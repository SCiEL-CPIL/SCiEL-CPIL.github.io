import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x*y
    # 4: x^2
    # 5: y^2
    # 6: sin(x)
    # 7: sin(y)
    # 8: cos(x)
    # 9: cos(y)
    
    # Calculate terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Trigonometric terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    
    # Vectorized computation
    pred = (params[0] * np.ones_like(x) + 
            params[1] * x + 
            params[2] * y + 
            params[3] * xy + 
            params[4] * x2 + 
            params[5] * y2 + 
            params[6] * sin_x + 
            params[7] * sin_y + 
            params[8] * cos_x + 
            params[9] * cos_y)
            
    return pred
