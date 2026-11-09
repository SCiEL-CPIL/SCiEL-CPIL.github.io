# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a Fourier series basis.
    
    Uses sine and cosine terms up to the 5th harmonic to capture
    periodic or oscillatory structures. The orthogonality of trigonometric
    functions on the domain ensures well-conditioned gradients for BFGS.
    
    Basis functions (10 total):
    0: Constant: 1
    1: sin(x)
    2: cos(x)
    3: sin(2x)
    4: cos(2x)
    5: sin(3x)
    6: cos(3x)
    7: sin(4x)
    8: cos(4x)
    9: sin(5x)
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute Fourier basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    sin_3x = np.sin(3 * x)
    cos_3x = np.cos(3 * x)
    sin_4x = np.sin(4 * x)
    cos_4x = np.cos(4 * x)
    sin_5x = np.sin(5 * x)
    
    # Combine all basis functions with their parameters
    result = (p0 * 1.0 + 
              p1 * sin_x + 
              p2 * cos_x + 
              p3 * sin_2x + 
              p4 * cos_2x + 
              p5 * sin_3x + 
              p6 * cos_3x + 
              p7 * sin_4x + 
              p8 * cos_4x + 
              p9 * sin_5x)
    
    return result
# EVOLVE-BLOCK-END
