"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Small Neural Network with 3 hidden tanh units.
    params:
    0: w1
    1: b1
    2: w2
    3: b2
    4: w3
    5: b3
    6: v1
    7: v2
    8: v3
    9: c (bias)
    
    f(x) = c + v1*tanh(w1*x + b1) + v2*tanh(w2*x + b2) + v3*tanh(w3*x + b3)
    """
    w1, b1, w2, b2, w3, b3, v1, v2, v3, c = params
    
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * x + b2)
    h3 = np.tanh(w3 * x + b3)
    
    result = c + v1 * h1 + v2 * h2 + v3 * h3
    
    return result
# EVOLVE-BLOCK-END
