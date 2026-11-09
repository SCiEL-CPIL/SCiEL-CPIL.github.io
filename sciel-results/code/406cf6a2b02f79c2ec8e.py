"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of basis functions for symbolic regression.
    The evaluator fits 10 parameters via BFGS starting from all ones.
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: x^5
    6: sin(x)
    7: cos(x)
    8: sin(2x)
    9: log(x)
    
    This set covers polynomial (up to degree 5), periodic, and logarithmic
    behaviors common in symbolic regression tasks.
    """
    x = np.asarray(x, dtype=np.float64)
    x_safe = np.clip(x, 1e-8, None)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = x ** 5
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.sin(2 * x)
    b9 = np.log(x_safe)
    
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
         
    return y

# EVOLVE-BLOCK-END
