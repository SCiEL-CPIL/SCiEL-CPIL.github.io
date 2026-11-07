"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Pure trigonometric (Fourier-like) basis to model periodic/oscillatory targets.
    
    The domain is [-1, 1]. We use frequencies scaled so that the basis functions
    have integer or half-integer periods relative to the domain length of 2.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: cos(pi*x)
    2: cos(pi*y)
    3: cos(2*pi*x)
    4: cos(2*pi*y)
    5: sin(pi*x)
    6: sin(pi*y)
    7: cos(pi*x)*cos(pi*y)
    8: sin(pi*x)*sin(pi*y)
    9: cos(pi*(x+y))
    
    This representation is well-conditioned because trigonometric functions are
    bounded and orthogonal on the interval (with appropriate weights). It avoids
    the ill-conditioning of high-degree polynomials and can capture oscillatory
    behavior that polynomials cannot.
    """
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Precompute common terms
    pi_x = np.pi * x
    pi_y = np.pi * y
    
    # Basis terms
    t0 = np.ones_like(x)
    t1 = np.cos(pi_x)
    t2 = np.cos(pi_y)
    t3 = np.cos(2.0 * pi_x)
    t4 = np.cos(2.0 * pi_y)
    t5 = np.sin(pi_x)
    t6 = np.sin(pi_y)
    t7 = np.cos(pi_x) * np.cos(pi_y)
    t8 = np.sin(pi_x) * np.sin(pi_y)
    t9 = np.cos(pi_x + pi_y)
    
    # Linear combination
    pred = (params[0] * t0 +
            params[1] * t1 +
            params[2] * t2 +
            params[3] * t3 +
            params[4] * t4 +
            params[5] * t5 +
            params[6] * t6 +
            params[7] * t7 +
            params[8] * t8 +
            params[9] * t9)
    
    return pred
# EVOLVE-BLOCK-END
