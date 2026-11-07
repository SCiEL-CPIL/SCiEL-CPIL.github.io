"""Fourier series expansion for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a richer Fourier basis in x and y.
    params: array of length 10, used as coefficients for the basis functions.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: sin(x)
    2: sin(y)
    3: cos(x)
    4: cos(y)
    5: sin(x)*sin(y)
    6: cos(x)*cos(y)
    7: sin(x+y)
    8: cos(x+y)
    9: sin(x-y)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trig arguments
    cx = np.cos(x)
    sx = np.sin(x)
    cy = np.cos(y)
    sy = np.sin(y)
    
    sxy_prod = sx * sy
    cxy_prod = cx * cy
    
    sxy_plus = np.sin(x + y)
    cxy_plus = np.cos(x + y)
    sxy_minus = np.sin(x - y)
    
    # Combine with parameters
    result = (
        p0
        + p1 * sx
        + p2 * sy
        + p3 * cx
        + p4 * cy
        + p5 * sxy_prod
        + p6 * cxy_prod
        + p7 * sxy_plus
        + p8 * cxy_plus
        + p9 * sxy_minus
    )
    
    return result
# EVOLVE-BLOCK-END
