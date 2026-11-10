"""Symbolic regression with a rich basis and multi-start BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _build_basis(x):
    """
    Build a basis matrix of shape (n, 10) with diverse functions.
    Basis:
      0: 1
      1: x
      2: x^2
      3: x^3
      4: x^4
      5: sin(x)
      6: cos(x)
      7: tanh(x)
      8: exp(-x^2)
      9: exp(-x^2/4)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    B = np.empty((n, 10), dtype=np.float64)
    
    B[:, 0] = 1.0
    B[:, 1] = x
    B[:, 2] = x * x
    B[:, 3] = B[:, 2] * x
    B[:, 4] = B[:, 2] * B[:, 2]
    B[:, 5] = np.sin(x)
    B[:, 6] = np.cos(x)
    B[:, 7] = np.tanh(x)
    x2 = x * x
    B[:, 8] = np.exp(-x2)
    B[:, 9] = np.exp(-x2 / 4.0)
    
    return B


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use the first 10 params as coefficients for a fixed basis of 10 functions.
    This is a linear model in params, so BFGS will converge quickly to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    B = _build_basis(x)
    
    return B @ p
# EVOLVE-BLOCK-END
