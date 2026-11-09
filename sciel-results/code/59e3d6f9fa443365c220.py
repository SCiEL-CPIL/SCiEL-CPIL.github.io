import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Row 1 (Top): 4 squares, side 0.25
    # y_center = 1 - 0.125 = 0.875? 
    # Let's stack from bottom to top to be safe.
    
    # Row 3 (Bottom): 3 squares, side 1/3
    s3 = 1.0 / 3.0
    y3 = s3 / 2.0
    x3 = np.array([1.0/6.0, 1.0/2.0, 5.0/6.0])
    centers[0:3, 0] = x3
    centers[0:3, 1] = y3
    sides[0:3] = s3
    
    # Row 2 (Middle): 3 squares, side 1/3
    s2 = 1.0 / 3.0
    y2 = s3 + s2 / 2.0  # 1/3 + 1/6 = 1/2
    x2 = np.array([1.0/6.0, 1.0/2.0, 5.0/6.0])
    centers[3:6, 0] = x2
    centers[3:6, 1] = y2
    sides[3:6] = s2
    
    # Row 1 (Top): 4 squares, side 0.25
    s1 = 0.25
    y1 = s3 + s2 + s1 / 2.0 # 2/3 + 1/8 = 16/24 + 3/24 = 19/24
    x1 = np.array([0.125, 0.375, 0.625, 0.875])
    centers[6:10, 0] = x1
    centers[6:10, 1] = y1
    sides[6:10] = s1
    
    return centers, sides
