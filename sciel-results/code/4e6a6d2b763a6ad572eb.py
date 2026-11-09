import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute features
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Trigonometric terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    
    # Basis:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: xy
    # 6: sin(x)
    # 7: sin(y)
    # 8: cos(x)
    # 9: cos(y)
    
    pred = (p0 + 
            p1 * x + 
            p2 * y + 
            p3 * x2 + 
            p4 * y2 + 
            p5 * xy + 
            p6 * sin_x + 
            p7 * sin_y + 
            p8 * cos_x + 
            p9 * cos_y)
            
    return pred
