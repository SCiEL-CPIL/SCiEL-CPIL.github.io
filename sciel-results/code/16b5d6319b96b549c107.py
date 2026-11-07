"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(1, 9):
        T[n+1, :] = 2.0 * t * T[n, :] - T[n-1, :]
    
    # Combine with parameters
    result = np.zeros_like(x)
    for i in range(10):
        result += params[i] * T[i, :]
    
    return result
# EVOLVE-BLOCK-END
