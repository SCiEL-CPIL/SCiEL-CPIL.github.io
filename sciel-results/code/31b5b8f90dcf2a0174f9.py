import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining low-degree polynomials and trigonometric functions.
    This spans a richer space than pure polynomials, potentially capturing
    periodic or oscillatory components of the hidden function while retaining
    polynomial flexibility.
    
    Basis Functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: xy
    6: sin(x)
    7: sin(y)
    8: cos(x)
    9: cos(y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = np.sin(x)
    b7 = np.sin(y)
    b8 = np.cos(x)
    b9 = np.cos(y)
    
    # Linear combination
    pred = (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
            
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
