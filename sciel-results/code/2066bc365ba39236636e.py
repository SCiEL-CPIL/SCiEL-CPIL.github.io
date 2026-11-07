# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Fourier product basis in x and y.
    params: array of length 10, used as coefficients for the basis functions.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: cos(x)
    2: sin(x)
    3: cos(y)
    4: sin(y)
    5: cos(x) * cos(y)
    6: cos(x) * sin(y)
    7: sin(x) * cos(y)
    8: sin(x) * sin(y)
    9: cos(2*x)  -- Second harmonic in x
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trig arguments
    cx = np.cos(x)
    sx = np.sin(x)
    cy = np.cos(y)
    sy = np.sin(y)
    
    # Product terms
    cx_cy = cx * cy
    cx_sy = cx * sy
    sx_cy = sx * cy
    sx_sy = sx * sy
    
    # Higher harmonic
    c2x = np.cos(2 * x)
    
    # Combine with parameters
    result = (
        p0
        + p1 * cx
        + p2 * sx
        + p3 * cy
        + p4 * sy
        + p5 * cx_cy
        + p6 * cx_sy
        + p7 * sx_cy
        + p8 * sx_sy
        + p9 * c2x
    )
    
    return result
# EVOLVE-BLOCK-END
