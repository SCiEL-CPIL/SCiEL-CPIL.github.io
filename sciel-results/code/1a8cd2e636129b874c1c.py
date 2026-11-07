"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a fixed basis of linear and tanh terms.
    The basis is chosen to be well-conditioned for BFGS optimization from all-ones
    start, avoiding the divergence issues of high-degree polynomials.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: tanh(3*x)
    4: tanh(3*y)
    5: tanh(3*(x+y))
    6: tanh(3*(x-y))
    7: tanh(5*x)
    8: tanh(5*y)
    9: tanh(5*(x+y))
    """
    p = params
    
    # Pre-compute tanh features
    # Scaling factors 3 and 5 help capture non-linearities within [-1, 1]
    # without being too sensitive to noise, while providing curvature.
    
    t1 = np.tanh(3.0 * x)
    t2 = np.tanh(3.0 * y)
    t3 = np.tanh(3.0 * (x + y))
    t4 = np.tanh(3.0 * (x - y))
    
    t5 = np.tanh(5.0 * x)
    t6 = np.tanh(5.0 * y)
    t7 = np.tanh(5.0 * (x + y))
    
    # Linear combination of features
    # p[0] is the bias/intercept
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * t1 +
        p[4] * t2 +
        p[5] * t3 +
        p[6] * t4 +
        p[7] * t5 +
        p[8] * t6 +
        p[9] * t7
    )
    
    return pred
# EVOLVE-BLOCK-END
