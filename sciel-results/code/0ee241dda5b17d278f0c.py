"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis of low-order polynomials 
    and first-order Fourier terms.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: sin(x)
    7: cos(x)
    8: sin(y)
    9: cos(y)
    
    This basis is linear in parameters, ensuring the least squares problem is convex.
    The inclusion of linear terms first helps BFGS converge quickly from the 
    all-ones initialization, while the Fourier terms allow for smooth non-linear 
    corrections if the target function exhibits periodic behavior or curvature 
    that polynomials might fit less efficiently in this range.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Compute basis terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    
    # Combine with parameters
    pred = (params[0] 
            + params[1] * x 
            + params[2] * y 
            + params[3] * x2 
            + params[4] * y2 
            + params[5] * xy 
            + params[6] * sin_x 
            + params[7] * cos_x 
            + params[8] * sin_y 
            + params[9] * cos_y)
            
    return pred
# EVOLVE-BLOCK-END
