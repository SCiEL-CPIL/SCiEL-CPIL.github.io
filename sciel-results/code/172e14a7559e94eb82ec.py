"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Chebyshev polynomial basis of degree 9 for well-conditioned,
    # smooth, and expressive representation.
    # params[0:10] are the coefficients for T_0 through T_9.
    
    # Clip x to [-1, 1] to avoid numerical issues with Chebyshev polynomials
    # outside the standard domain. Since x is in [-3, 3], we scale it.
    # Actually, Chebyshev polynomials are defined for all real x, but they
    # grow rapidly outside [-1,1]. Let's use a scaled version.
    
    # Map x from [-3, 3] to [-1, 1] for better conditioning
    x_scaled = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    
    T = np.zeros((10, len(x_scaled)))
    T[0, :] = 1.0
    if len(x_scaled) > 0:
        T[1, :] = x_scaled
        for n in range(2, 10):
            T[n, :] = 2.0 * x_scaled * T[n-1, :] - T[n-2, :]
    
    # Linear combination: y = sum_{k=0}^{9} params[k] * T_k(x/3)
    # T is shape (10, N), params is shape (10,)
    # We want (N,) output: T.T @ params
    return T.T @ params
# EVOLVE-BLOCK-END
