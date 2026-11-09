"""Symbolic regression with mixed polynomial and reciprocal basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis with linear, interaction, and reciprocal terms.
    
    Uses 10 parameters arranged as:
    - params[0]: constant
    - params[1]: x
    - params[2]: y
    - params[3]: z
    - params[4]: x*y
    - params[5]: x*z
    - params[6]: y*z
    - params[7]: 1/x
    - params[8]: 1/y
    - params[9]: 1/z
    """
    p = params
    
    # Interaction terms and reciprocal terms
    # This structure allows the model to capture multiplicative dependencies
    # and inverse dependencies simultaneously.
    
    result = (p[0] * 1.0
              + p[1] * x
              + p[2] * y
              + p[3] * z
              + p[4] * (x * y)
              + p[5] * (x * z)
              + p[6] * (y * z)
              + p[7] * (1.0 / x)
              + p[8] * (1.0 / y)
              + p[9] * (1.0 / z))
    
    return result
# EVOLVE-BLOCK-END
