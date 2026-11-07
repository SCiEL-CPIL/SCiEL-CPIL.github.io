"""Symbolic regression with a well-conditioned, rich fixed basis."""
# EVOLVE-BLOCK-START
import numpy as np


def _cheb(x, n):
    """Compute Chebyshev polynomials T_0(x)..T_n(x) via recurrence on [-1,1]."""
    c = np.zeros((x.size, n + 1), dtype=np.float64)
    c[:, 0] = 1.0
    if n >= 1:
        c[:, 1] = x
    for k in range(2, n + 1):
        c[:, k] = 2.0 * x * c[:, k - 1] - c[:, k - 2]
    return c


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter linear model with a fixed, well-conditioned basis on [-2, 2].
    
    Basis design:
      - Chebyshev polynomials T_0..T_4 (mapped x -> x/2) for smooth global trends
      - Gaussian bumps at different centers/scales for local features
      - tanh non-linearities for saturation effects
      
    This gives a rich but well-conditioned basis that BFGS can fit reliably.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    
    # Map x from [-2,2] to [-1,1] for Chebyshev
    xm = x / 2.0
    
    # Chebyshev basis: T_0, T_1, T_2, T_3, T_4 (5 params)
    c = _cheb(xm, 4)
    b0 = c[:, 0]  # 1
    b1 = c[:, 1]  # xm
    b2 = c[:, 2]  # 2*xm^2 - 1
    b3 = c[:, 3]  # 4*xm^3 - 3*xm
    b4 = c[:, 4]  # 8*xm^4 - 8*xm^2 + 1
    
    # Gaussian bumps for local features (3 params)
    # Centers at -1, 0, 1 with different widths
    b5 = np.exp(-((x - (-1.0)) ** 2) / 0.8)
    b6 = np.exp(-((x - 0.0) ** 2) / 0.8)
    b7 = np.exp(-((x - 1.0) ** 2) / 0.8)
    
    # tanh non-linearities for saturation (2 params)
    b8 = np.tanh(x)
    b9 = np.tanh(2.0 * x)
    
    # Linear combination
    y = (params[0] * b0 + params[1] * b1 + params[2] * b2 +
         params[3] * b3 + params[4] * b4 +
         params[5] * b5 + params[6] * b6 + params[7] * b7 +
         params[8] * b8 + params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
