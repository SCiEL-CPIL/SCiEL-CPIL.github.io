"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining polynomial basis for global structure with
    tanh and RBF terms for local flexibility and non-linearity.
    
    Basis terms:
    0: 1 (bias)
    1: x
    2: y
    3: x^2
    4: y^2
    5: xy
    6: tanh(x)
    7: tanh(y)
    8: RBF at (1, 1)
    9: RBF at (-1, -1)
    """
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Polynomial terms for global structure
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Bounded tanh terms for smooth non-linearity
    t1 = np.tanh(x)
    t2 = np.tanh(y)
    
    # RBF terms at opposite corners for local flexibility
    sigma = 0.7
    r1 = np.sqrt((x - 1.0)**2 + (y - 1.0)**2)
    r2 = np.sqrt((x + 1.0)**2 + (y + 1.0)**2)
    
    g1 = np.exp(-r1**2 / (2 * sigma**2))
    g2 = np.exp(-r2**2 / (2 * sigma**2))
    
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * x2 +
        p[4] * y2 +
        p[5] * xy +
        p[6] * t1 +
        p[7] * t2 +
        p[8] * g1 +
        p[9] * g2
    )
    return pred
# EVOLVE-BLOCK-END
