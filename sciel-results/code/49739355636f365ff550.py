import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Orthonormal Legendre polynomial basis for symbolic regression.
    Uses Chebyshev polynomials of the first kind, which are better conditioned
    on [-1,1] and have the equioscillation property for uniform error distribution.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / 3.0  # Map to [-1, 1]
    
    n = 10
    # Evaluate Chebyshev polynomials T_0 through T_9 on t
    # T_0(t) = 1, T_1(t) = t, T_{k}(t) = 2t*T_{k-1}(t) - T_{k-2}(t)
    P = np.zeros((len(t), n), dtype=np.float64)
    P[:, 0] = 1.0
    if n > 1:
        P[:, 1] = t
    for k in range(2, n):
        P[:, k] = 2.0 * t * P[:, k-1] - P[:, k-2]
    
    # Normalize columns for orthonormality on [-1,1]
    # ||T_0||^2 = 1, ||T_k||^2 = 1/2 for k>=1
    # So normalize: T_0 stays, T_k * sqrt(2) for k>=1
    P[:, 0] *= 1.0
    if n > 1:
        P[:, 1:] *= np.sqrt(2.0)
    
    return P @ params
