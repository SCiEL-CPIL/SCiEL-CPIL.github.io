"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(1, 9):
        T[n+1, :] = 2.0 * t * T[n, :] - T[n-1, :]
    
    # Normalize each basis function to have unit L2 norm over [-2,2]
    # This makes the optimization landscape more isotropic for BFGS
    # Expected L2 norm: sqrt(E[T_n(t)^2]) where t ~ Uniform[-1,1]
    # For Chebyshev: E[T_0^2] = 1, E[T_n^2] = 1/2 for n >= 1
    norms = np.ones(10)
    norms[0] = 1.0  # T_0 = 1, norm = 1
    norms[1:] = np.sqrt(0.5)  # E[T_n^2] = 1/2 for n>=1
    # Scale so that each basis function has unit expected norm
    # Divide by the norm to get unit-norm basis
    T = T / norms[:, np.newaxis]
    
    # Combine with parameters
    result = np.tensordot(params, T, axes=(0, 0))
    
    return result
# EVOLVE-BLOCK-END
