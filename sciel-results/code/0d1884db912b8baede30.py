import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a richer basis that includes higher-order terms and interactions
    # to better capture potential nonlinearities in the hidden function.
    # Basis:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x*y
    # 4: x^2
    # 5: y^2
    # 6: x^3
    # 7: y^3
    # 8: x^2*y
    # 9: x*y^2
    
    # Calculate terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Vectorized computation
    pred = (params[0] * np.ones_like(x) + 
            params[1] * x + 
            params[2] * y + 
            params[3] * xy + 
            params[4] * x2 + 
            params[5] * y2 + 
            params[6] * x3 + 
            params[7] * y3 + 
            params[8] * x2y + 
            params[9] * xy2)
            
    return pred
