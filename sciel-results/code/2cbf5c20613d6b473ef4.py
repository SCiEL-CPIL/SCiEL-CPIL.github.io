"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a Chebyshev polynomial basis of degree 9.
    The domain x is [-3, 3]. We scale x to u in [-1, 1] to use standard Chebyshev polynomials.
    params[0] corresponds to T_0(u) = 1
    params[1] corresponds to T_1(u) = u
    ...
    params[9] corresponds to T_9(u)
    """
    # Scale x to [-1, 1]
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_k(u) for k=0..9
    # T_0(u) = 1
    # T_1(u) = u
    # T_{k+1}(u) = 2*u*T_k(u) - T_{k-1}(u)
    
    # Initialize basis vectors
    t_k_minus_1 = np.ones_like(u)  # T_0
    t_k = u.copy()                  # T_1
    
    # Store results in a list of arrays or a matrix
    # We need 10 coefficients, so T_0 to T_9
    basis = np.empty((len(u), 10), dtype=np.float64)
    basis[:, 0] = t_k_minus_1
    basis[:, 1] = t_k
    
    for k in range(2, 10):
        t_k_plus_1 = 2.0 * u * t_k - t_k_minus_1
        basis[:, k] = t_k_plus_1
        t_k_minus_1 = t_k
        t_k = t_k_plus_1
        
    # Prediction is linear combination of basis functions
    # y = sum(params[i] * basis[:, i])
    return basis @ params
# EVOLVE-BLOCK-END
