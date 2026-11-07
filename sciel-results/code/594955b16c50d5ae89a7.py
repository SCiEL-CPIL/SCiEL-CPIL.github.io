"""Symbolic regression with Chebyshev polynomial basis for better conditioning."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 for symbolic regression.
    
    params (length 10): coefficients for Chebyshev polynomials T_0 through T_9.
    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
    
    The domain is [-2, 2]. Chebyshev polynomials are well-conditioned on [-1, 1],
    so we scale x to [-1, 1] by dividing by 2.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0 = 1, T_1 = t
    # T_n = 2*t*T_{n-1} - T_{n-2}
    
    n = len(x)
    basis = np.empty((n, 10), dtype=np.float64)
    
    basis[:, 0] = 1.0  # T_0
    if n > 0:
        basis[:, 1] = t  # T_1
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Linear combination with params
    result = basis @ params
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
