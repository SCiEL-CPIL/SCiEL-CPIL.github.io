"""Improved symbolic regression with a richer, more flexible basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a carefully chosen basis of 10 functions.
    
    The key insight from the cell findings: spreading basis functions across
    the domain helps BFGS fit the global shape immediately. Instead of pure
    Chebyshev polynomials (which are global), we mix in localized "bump"
    functions and a variety of global functions to give the optimizer more
    degrees of freedom.
    
    Basis functions (all defined on x in [-2, 2]):
    0: 1 (constant)
    1: x (linear)
    2: x^2 (quadratic)
    3: sin(x) 
    4: cos(x)
    5: exp(-x^2/2) - Gaussian bump centered at 0
    6: exp(-(x-1)^2/2) - Gaussian bump centered at 1
    7: exp(-(x+1)^2/2) - Gaussian bump centered at -1
    8: 1/(1+x^2) - Lorentzian
    9: |x| - absolute value (non-smooth but useful)
    
    This mix gives BFGS:
    - Global polynomial/trig terms for overall shape
    - Localized bumps to capture features
    - Non-smooth term for kinks
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = np.sin(x)
    b4 = np.cos(x)
    b5 = np.exp(-x ** 2 / 2.0)
    b6 = np.exp(-(x - 1.0) ** 2 / 2.0)
    b7 = np.exp(-(x + 1.0) ** 2 / 2.0)
    b8 = 1.0 / (1.0 + x ** 2)
    b9 = np.abs(x)
    
    # Linear combination
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
