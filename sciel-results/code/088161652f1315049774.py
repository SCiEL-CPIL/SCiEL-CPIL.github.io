import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    return (p[0] + 
            p[1] * x + 
            p[2] * v + 
            p[3] * (x * v) + 
            p[4] * x3 + 
            p[5] * v3 + 
            p[6] * (x2 * v) + 
            p[7] * (x * v2) + 
            p[8] * x4 + 
            p[9] * v4)
