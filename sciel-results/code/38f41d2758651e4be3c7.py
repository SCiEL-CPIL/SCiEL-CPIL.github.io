# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Scaled monomial polynomial approximation of degree 9 on [0.1, 6].
    Uses centered and scaled x to improve conditioning for BFGS.
    y = p0 + p1*u + p2*u^2 + ... + p9*u^9, where u = (x - 3.05)/2.95
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Center and scale x to [-1, 1]
    u = (x - 3.05) / 2.95
    
    # Build monomial basis: 1, u, u^2, ..., u^9
    N = x.shape[0]
    if N == 0:
        return np.array([], dtype=np.float64)
    
    # Compute powers efficiently
    basis = np.empty((10, N), dtype=np.float64)
    basis[0, :] = 1.0
    basis[1, :] = u
    for k in range(2, 10):
        basis[k, :] = basis[k-1, :] * u
    
    # Linear combination
    y = p @ basis
    
    return y
# EVOLVE-BLOCK-END
