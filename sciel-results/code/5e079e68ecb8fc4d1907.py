"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Well-conditioned basis of 10 smooth functions.
    
    Basis functions:
    0: 1          (constant)
    1: x          (linear)
    2: x^2        (quadratic)
    3: x^3        (cubic)
    4: x^4        (quartic)
    5: exp(-x^2)  (Gaussian bump)
    6: exp(-x^2/2) (wider Gaussian)
    7: tanh(x)    (smooth sigmoid)
    8: x*tanh(x)  (even sigmoid)
    9: sin(x)     (single periodic term)
    """
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = b2 * b2
    
    b5 = np.exp(-b2)
    b6 = np.exp(-b2 / 2.0)
    b7 = np.tanh(x)
    b8 = x * b7
    b9 = np.sin(x)
    
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
