"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-term Fourier series with fixed frequencies {1, 2, 3, 4, 5}.
    Basis: [1, sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x), sin(4x), cos(4x), sin(5x), cos(5x)]
    Wait, that's 11 terms. We need exactly 10 params.
    
    Let's use:
    params[0] * 1
    params[1] * sin(x)
    params[2] * cos(x)
    params[3] * sin(2*x)
    params[4] * cos(2*x)
    params[5] * sin(3*x)
    params[6] * cos(3*x)
    params[7] * sin(4*x)
    params[8] * cos(4*x)
    params[9] * sin(5*x)
    
    This gives 10 terms. We drop cos(5x) to keep it at 10 params.
    Alternatively, we could do 5 sines and 5 cosines without the constant? 
    But a constant term is important for capturing the mean.
    
    Let's stick with: constant, sin(1), cos(1), sin(2), cos(2), sin(3), cos(3), sin(4), cos(4), sin(5).
    This is a valid linear-in-params basis.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is 1D for safety, though it should already be
    if x.ndim != 1:
        x = x.flatten()
        
    # Compute basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    sin_3x = np.sin(3 * x)
    cos_3x = np.cos(3 * x)
    sin_4x = np.sin(4 * x)
    cos_4x = np.cos(4 * x)
    sin_5x = np.sin(5 * x)
    
    # Combine with parameters
    # params[0] is the constant term
    result = (
        params[0] +
        params[1] * sin_x +
        params[2] * cos_x +
        params[3] * sin_2x +
        params[4] * cos_2x +
        params[5] * sin_3x +
        params[6] * cos_3x +
        params[7] * sin_4x +
        params[8] * cos_4x +
        params[9] * sin_5x
    )
    
    return result
# EVOLVE-BLOCK-END
