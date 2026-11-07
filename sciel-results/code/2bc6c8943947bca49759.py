"""Rewritten symbolic regression with a richer, well-conditioned basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a rich basis of polynomial, trigonometric,
    and rational-like terms. Designed to be well-conditioned for BFGS starting
    from all-ones.
    
    Basis (10 terms):
    b0 = 1
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    b5 = sin(x * 1.0)
    b6 = cos(x * 1.0)
    b7 = sin(x * 2.0)
    b8 = cos(x * 2.0)
    b9 = 1.0 / (1.0 + x)
    
    This mix captures polynomial trends, periodic components, and a decaying
    rational component that can model many common smooth functions.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = params
    
    # Polynomial basis
    b0 = 1.0
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    
    # Exponential decay basis - captures rapid changes
    b5 = np.exp(-x)
    b6 = np.exp(-0.5 * x)
    
    # Trigonometric basis
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # Logarithmic basis
    b9 = np.log1p(x)
    
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
