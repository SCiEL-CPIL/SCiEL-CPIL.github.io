"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a Chebyshev polynomial basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2x - (max + min)) / (max - min)
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Number of Chebyshev terms
    n_terms = 10
    
    # Compute Chebyshev polynomials T_k(t) for k=0..9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)
    
    basis = np.empty((len(x), n_terms), dtype=np.float64)
    
    if n_terms > 0:
        basis[:, 0] = 1.0
        
    if n_terms > 1:
        basis[:, 1] = t
        
    for k in range(2, n_terms):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Prediction is the linear combination of the basis functions
    return basis @ params
# EVOLVE-BLOCK-END
