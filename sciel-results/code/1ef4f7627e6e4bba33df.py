"""Symbolic regression with diverse basis functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute y = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Basis functions chosen to span common functional forms on [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure we have at least 10 params
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), constant_values=0.0)
    
    # Define 10 basis functions
    b0 = np.ones_like(x)           # constant
    b1 = x                          # linear
    b2 = x ** 2                     # quadratic
    b3 = x ** 3                     # cubic
    b4 = x ** 4                     # quartic
    b5 = np.sin(x)                  # sine
    b6 = np.cos(x)                  # cosine
    b7 = np.exp(x)                  # exponential growth
    b8 = np.exp(-x)                 # exponential decay
    b9 = np.tanh(x)                 # hyperbolic tangent
    
    # Linear combination
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
              p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
