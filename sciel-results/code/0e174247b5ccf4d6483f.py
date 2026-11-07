"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Use recurrence for numerical stability
    # T_0 = 1, T_1 = t, T_k = 2*t*T_{k-1} - T_{k-2}
    T = np.empty((10, len(t)), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
    
    # Linear combination of the basis functions
    # This is linear in params, ensuring robust convergence of the optimizer
    result = np.zeros_like(t)
    for k in range(10):
        result += params[k] * T[k]
    return result
# EVOLVE-BLOCK-END
