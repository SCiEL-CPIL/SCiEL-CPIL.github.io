# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Pure Fourier/Tensor-Product basis.
    Explores periodic structure with higher frequencies and interaction terms.
    
    Basis functions:
    1. Constant
    2. sin(pi*x)
    3. cos(pi*x)
    4. sin(pi*y)
    5. cos(pi*y)
    6. sin(2*pi*x)
    7. cos(2*pi*x)
    8. sin(2*pi*y)
    9. cos(2*pi*y)
    10. sin(pi*x)*sin(pi*y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Precompute trig terms
    sin_px = np.sin(np.pi * x)
    cos_px = np.cos(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_py = np.cos(np.pi * y)
    
    sin_2px = np.sin(2 * np.pi * x)
    cos_2px = np.cos(2 * np.pi * x)
    sin_2py = np.sin(2 * np.pi * y)
    cos_2py = np.cos(2 * np.pi * y)
    
    b0 = np.ones_like(x)
    b1 = sin_px
    b2 = cos_px
    b3 = sin_py
    b4 = cos_py
    b5 = sin_2px
    b6 = cos_2px
    b7 = sin_2py
    b8 = cos_2py
    b9 = sin_px * sin_py
    
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
