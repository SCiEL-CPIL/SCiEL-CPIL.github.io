"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid polynomial-trigonometric basis.
    This basis is designed to capture both smooth non-periodic trends (via polynomials)
    and periodic behaviors (via trigonometric functions), which are common in symbolic regression tasks.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: xy
    5: y^2
    6: sin(x)
    7: sin(y)
    8: cos(x)
    9: cos(y)
    """
    p = params
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Trigonometric terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    
    # Linear combination
    result = (p[0] * 1.0 + 
              p[1] * x + 
              p[2] * y + 
              p[3] * x2 + 
              p[4] * xy + 
              p[5] * y2 + 
              p[6] * sin_x + 
              p[7] * sin_y + 
              p[8] * cos_x + 
              p[9] * cos_y)
    
    return result
# EVOLVE-BLOCK-END
