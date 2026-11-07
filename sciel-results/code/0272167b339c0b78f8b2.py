"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Composite model: 5th-degree polynomial + 4-term Fourier series.
    params[0:6] are coefficients for x^0 through x^5.
    params[6:10] are coefficients for sin(x), cos(x), sin(2x), cos(2x).
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial part (degree 5)
    # Using Horner's method for numerical stability and speed
    p = np.zeros_like(x)
    for i in range(5, -1, -1):
        p = p * x + params[i]
        
    # Fourier part (frequencies 1 and 2)
    # params[6]: sin(x), params[7]: cos(x), params[8]: sin(2x), params[9]: cos(2x)
    fourier = (params[6] * np.sin(x) + 
               params[7] * np.cos(x) + 
               params[8] * np.sin(2.0 * x) + 
               params[9] * np.cos(2.0 * x))
               
    return p + fourier
# EVOLVE-BLOCK-END
