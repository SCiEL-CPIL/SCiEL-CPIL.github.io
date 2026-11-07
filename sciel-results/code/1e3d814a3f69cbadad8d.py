"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for Legendre polynomials
    # Legendre polynomials are L2-orthogonal with respect to the uniform distribution,
    # which matches the MSE objective and the uniform sampling of x.
    t = x / 2.0
    
    # Compute Legendre polynomials P_0 through P_9
    # P_0(t) = 1
    # P_1(t) = t
    # (n+1)P_{n+1}(t) = (2n+1)t*P_n(t) - n*P_{n-1}(t)
    
    P = np.zeros((10, len(x)))
    P[0, :] = 1.0
    P[1, :] = t
    
    for n in range(1, 9):
        # P_{n+1} = ((2n+1)t*P_n - n*P_{n-1}) / (n+1)
        P[n+1, :] = ((2.0 * n + 1.0) * t * P[n, :] - n * P[n-1, :]) / (n + 1.0)
    
    # Combine with parameters
    result = np.zeros_like(x)
    for i in range(10):
        result += params[i] * P[i, :]
    
    return result
# EVOLVE-BLOCK-END
