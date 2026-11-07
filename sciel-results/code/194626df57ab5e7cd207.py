"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6] to [-1, 1]
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Precompute Legendre polynomials P_0 to P_9
    # P_0(t) = 1
    # P_1(t) = t
    # (n+1)P_{n+1}(t) = (2n+1)tP_n(t) - nP_{n-1}(t)
    
    n = len(x)
    P = np.zeros((n, 10), dtype=np.float64)
    P[:, 0] = 1.0
    P[:, 1] = t
    
    for k in range(2, 10):
        P[:, k] = ((2.0 * k - 1.0) * t * P[:, k-1] - (k - 1.0) * P[:, k-2]) / k
        
    # Compute linear combination
    return P @ params
# EVOLVE-BLOCK-END
