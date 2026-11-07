"""Symbolic regression with 1 input using a well-conditioned basis + internal refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a standardized Chebyshev polynomial basis (degree 0-9) mapped to [0.1, 6].
    The basis is well-conditioned for BFGS. We do NOT apply tanh to params,
    keeping the model linear in params so the least-squares solution is reachable.
    
    Key structural elements:
    1. Chebyshev basis on t = (2x - 6.1)/5.9, clipped to [-1, 1].
    2. Each column standardized: divide by sqrt(0.5) for k>=1 (variance of T_k for t~U[-1,1]).
    3. Linear model: pred = basis_std @ params.
    
    This is a pure linear model in params, so BFGS from all-ones should converge
    to the global optimum (the least-squares solution) reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)
    
    n_points = x.shape[0]
    
    # Compute Chebyshev polynomials T_0 to T_9
    basis = np.empty((n_points, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    if n_points > 0:
        basis[:, 1] = t
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Standardize each basis column using precomputed moments for uniform t in [-1, 1].
    # For T_k(t) with t ~ Uniform[-1, 1]:
    #   mean = 0 for all k (by symmetry)
    #   var(T_0) = 1
    #   var(T_k) = 0.5 for k >= 1
    std = np.ones(10, dtype=np.float64)
    std[1:] = np.sqrt(0.5)
    
    # Standardize the basis matrix
    basis_std = basis / std
    
    # Linear combination
    return basis_std @ params
# EVOLVE-BLOCK-END
