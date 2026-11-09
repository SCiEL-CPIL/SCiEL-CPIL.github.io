import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Check for non-finite inputs
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return np.zeros_like(x)
        
    # Structural Change: Cubic Polynomial Basis
    # This provides a smooth, global approximation capability.
    # Terms: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2
    # This uses all 10 parameters efficiently.
    
    # Precompute powers to speed up evaluation
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Construct the basis matrix implicitly via linear combination
    # p0: 1
    # p1: x
    # p2: y
    # p3: x^2
    # p4: y^2
    # p5: xy
    # p6: x^3
    # p7: y^3
    # p8: x^2 y
    # p9: x y^2
    
    pred = (params[0] * 1.0 + 
            params[1] * x + 
            params[2] * y + 
            params[3] * x2 + 
            params[4] * y2 + 
            params[5] * xy + 
            params[6] * x3 + 
            params[7] * y3 + 
            params[8] * x2y + 
            params[9] * xy2)
            
    return pred
