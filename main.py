from ultralytics import YOLO
import cv2
import torch
import numpy as np

model_path = 'models/seg_model.pt'


model = YOLO(model_path)


#print('Model loaded successfully:', model)


# load image


image_path = 'asset/image2.jpg'


frame = cv2.imread(image_path)


# run model


results = model(frame)


#print('Raw results:', results)


for r in results:
    if r.boxes:
        for box in r.boxes:
            #get cordination of the bbx
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            # get the confident score and the class name
            cls = int(box.cls[0])
            conf = round(box.conf[0].item(),2)


            #print(f'Clss: {cls}, Confidence: {conf}, Bounding Box: {(x1, y1, x2, y2)}')






tensor = torch.tensor([0.85]) # A one element tensor


scalar = tensor[0].item()


#print(tensor, scalar)


#drawing the bounding box and labels
class_name = ['leftcurve', 'rightcurve', 'uturn']
for r in results:
    if r.boxes:
        for box in r.boxes:
            #get cordination of the bbx
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            # get the confident score and the class name
            cls = int(box.cls[0])
            conf = round(box.conf[0].item(),2)
            label = f'{class_name[cls]} {conf}'


            #drawing the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)


            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


# display the bounding box


#cv2.imshow('Detection', frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

for r in results:
    if r.masks:
        masks = r.masks.data.cpu().numpy()

        for mask in masks:
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            overly = np.zeros_like(frame, dtype=np.unit8)
            overly[mask_resized.astype(bool)] = [255, 0, 255]

            frame = cv2.addWeighted(frame, 0.80,  overly, 0.2, 0)

cv2.imshow('Segmentation', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()