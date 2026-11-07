"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a diverse, smooth mixed basis of functions.
    
    The evaluator fits the 10 params via BFGS. We provide a diverse set of
    basis functions that can approximate many common hidden functions
    (polynomials, trigonometric, hyperbolic, Gaussian bumps, etc.).
    
    Basis functions:
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: tanh(x)
    7: exp(-x^2)
    8: sin(2x)
    9: cos(2x)
    """
    x = np.asarray(x, dtype=np.float64)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = x * x * x
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.tanh(x)
    b7 = np.exp(-x * x)
    b8 = np.sin(2.0 * x)
    b9 = np.cos(2.0 * x)
    
    result = (params[0] * b0 +
              params[1] * b1 +
              params[2] * b2 +
              params[3] * b3 +
              params[4] * b4 +
              params[5] * b5 +
              params[6] * b6 +
              params[7] * b7 +
              params[8] * b8 +
              params[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
