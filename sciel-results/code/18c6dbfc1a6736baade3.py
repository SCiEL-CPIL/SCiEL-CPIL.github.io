"""Symbolic regression with Chebyshev basis for better conditioning."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis(x: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute Chebyshev polynomials of the first kind T_k(t) for k=0..n-1,
    where t = x/3 (mapping [-3,3] to [-1,1]).
    
    Returns shape (len(x), n).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    t = x / 3.0  # map to [-1, 1]
    
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.zeros((len(x), n))
    T[:, 0] = 1.0
    if n > 1:
        T[:, 1] = t
        for k in range(2, n):
            T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    return T


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a linear combination of shifted Chebyshev polynomials.
    params has length 10, one coefficient per basis function.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    basis = _chebyshev_basis(x, n=10)
    return basis @ params
# EVOLVE-BLOCK-END
