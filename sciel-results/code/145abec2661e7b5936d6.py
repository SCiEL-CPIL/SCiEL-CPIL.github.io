# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    Basis:
    1. x
    2. x^3
    3. x^5
    4. v
    5. v^3
    6. x^2 * v
    7. x * v^2
    8. x * v
    9. x^2
    10. bias
    """
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    
    v2 = v * v
    v3 = v2 * v
    
    x2v = x2 * v
    xv2 = x * v2
    xv = x * v
    
    return (params[0] * x +
            params[1] * x3 +
            params[2] * x5 +
            params[3] * v +
            params[4] * v3 +
            params[5] * x2v +
            params[6] * xv2 +
            params[7] * xv +
            params[8] * x2 +
            params[9])
# EVOLVE-BLOCK-END
