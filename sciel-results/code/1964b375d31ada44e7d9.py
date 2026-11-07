"""Combined symbolic regression program."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the hidden function using a mixed basis of polynomial and exponential terms.
    
    The linear model (score 1.0) is the baseline. The quadratic model (score 7.39) and
    current exponential model (score 6.38) are worse, suggesting the function is not
    a simple polynomial but likely involves exponential or other non-linear terms.
    
    Basis: 1, x, y, z, exp(x), exp(y), exp(z), x*y, x*z, y*z
    This captures exponential growth in each dimension plus interaction terms.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Compute basis elements
    ex = np.exp(x)
    ey = np.exp(y)
    ez = np.exp(z)
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Linear combination
    return c0 + c1 * x + c2 * y + c3 * z + c4 * ex + c5 * ey + c6 * ez + c7 * xy + c8 * xz + c9 * yz
# EVOLVE-BLOCK-END
