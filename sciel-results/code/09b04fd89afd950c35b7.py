"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier Series Approximation.
    Model: sum_{i=1}^{5} (a_i * cos(i*x) + b_i * sin(i*x))
    Parameters:
    0: a1, 1: b1
    2: a2, 3: b2
    4: a3, 5: b3
    6: a4, 7: b4
    8: a5, 9: b5
    """
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = params
    
    # Precompute trigonometric terms for efficiency
    # x is input array
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_2x = np.cos(2 * x)
    sin_2x = np.sin(2 * x)
    cos_3x = np.cos(3 * x)
    sin_3x = np.sin(3 * x)
    cos_4x = np.cos(4 * x)
    sin_4x = np.sin(4 * x)
    cos_5x = np.cos(5 * x)
    sin_5x = np.sin(5 * x)
    
    # Compute linear combination
    y = (a1 * cos_x + b1 * sin_x + 
         a2 * cos_2x + b2 * sin_2x + 
         a3 * cos_3x + b3 * sin_3x + 
         a4 * cos_4x + b4 * sin_4x + 
         a5 * cos_5x + b5 * sin_5x)
         
    return y
# EVOLVE-BLOCK-END
