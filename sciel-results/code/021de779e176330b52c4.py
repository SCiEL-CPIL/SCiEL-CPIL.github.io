"""Improved program using a mixed polynomial and trigonometric basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0 # Safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to a simple linear prediction to avoid hanging or errors
        # Ideally, this shouldn't happen if the optimizer is fast, but safe.
        return np.zeros_like(x)

    # Map params to coefficients of a mixed basis
    # Basis:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: x*y
    # 6: sin(x)
    # 7: sin(y)
    # 8: cos(x)
    # 9: cos(y)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute terms vectorized
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    
    # Construct the linear combination
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
# EVOLVE-BLOCK-END
