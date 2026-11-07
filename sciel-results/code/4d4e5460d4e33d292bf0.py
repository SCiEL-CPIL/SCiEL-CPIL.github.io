"""Symbolic regression with a diverse smooth basis for BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 diverse smooth basis functions.
    
    The evaluator fits params (length 10) via BFGS starting from all ones.
    We provide a well-conditioned, diverse basis that can approximate many
    common hidden functions: polynomials, trigonometric, Gaussian, sigmoid.
    
    Basis:
      0: 1
      1: x
      2: x^2
      3: x^3
      4: x^4
      5: sin(x)
      6: cos(x)
      7: exp(-x^2)
      8: tanh(x)
      9: x * exp(-x^2)  (odd Gaussian-like, captures asymmetric bumps)
    """
    x = np.asarray(x, dtype=np.float64)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = b2 * b2
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp(-b2)
    b8 = np.tanh(x)
    b9 = x * b7
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
