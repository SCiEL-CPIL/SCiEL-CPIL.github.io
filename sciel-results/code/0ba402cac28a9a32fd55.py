import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    p = np.asarray(params, dtype=np.float64)
    
    # 10 basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x ** 2
    b4 = y ** 2
    b5 = x * y
    b6 = x ** 3
    b7 = y ** 3
    b8 = np.sin(x)
    b9 = np.sin(y)
    
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + 
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 + 
            p[8] * b8 + p[9] * b9)
    
    return pred
