"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params length 10: use a rich basis with Chebyshev polynomials (well-conditioned)
    # plus some trigonometric terms to capture periodic behavior.
    # Basis:
    #   T_0..T_9 (Chebyshev of first kind on [-1,1], mapped from [-3,3])
    # We'll use all 10 slots for Chebyshev T_0 through T_9.
    
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    T = np.zeros((10, n), dtype=np.float64)
    T[0, :] = 1.0
    if n > 0:
        T[1, :] = t
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Transpose to get shape (n, 10)
    basis = T.T
    
    # params is length 10; clip to avoid overflow during optimization
    params = np.clip(np.asarray(params, dtype=np.float64), -1e6, 1e6)
    
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback: use only the first two terms which are always finite
        pred = np.clip(params[0], -1e6, 1e6) + np.clip(params[1], -1e6, 1e6) * t
    
    return pred
# EVOLVE-BLOCK-END
