"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for standard Legendre polynomials
    t = x / 2.0
    
    # Compute Legendre polynomials P_0 through P_9
    # P_0(t) = 1
    # P_1(t) = t
    # (n+1)P_{n+1}(t) = (2n+1)tP_n(t) - nP_{n-1}(t)
    
    P = np.zeros((10, len(x)))
    P[0, :] = 1.0
    P[1, :] = t
    
    for n in range(1, 9):
        P[n+1, :] = ((2*n + 1) * t * P[n, :] - n * P[n-1, :]) / (n + 1)
    
    # Combine with parameters
    result = np.zeros_like(x)
    for i in range(10):
        result += params[i] * P[i, :]
    
    return result
# EVOLVE-BLOCK-END
