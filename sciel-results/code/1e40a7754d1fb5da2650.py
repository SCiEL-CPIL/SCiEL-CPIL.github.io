"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: polynomial basis for global structure + tanh for bounded non-linearity.
    Basis terms:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: xy
    6: tanh(x)
    7: tanh(y)
    8: tanh(x+y)
    9: tanh(x-y)
    """
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Polynomial terms for global structure
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Bounded tanh terms for local flexibility and saturation
    t1 = np.tanh(x)
    t2 = np.tanh(y)
    t3 = np.tanh(x + y)
    t4 = np.tanh(x - y)
    
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * x2 +
        p[4] * y2 +
        p[5] * xy +
        p[6] * t1 +
        p[7] * t2 +
        p[8] * t3 +
        p[9] * t4
    )
    return pred
# EVOLVE-BLOCK-END
