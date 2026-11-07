"""Symbolic regression with standardized Chebyshev + auxiliary basis for BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Standardized basis function with 10 parameters.
    
    Uses Chebyshev polynomials (degree 0-6) mapped to [0.1, 6] as the primary basis,
    plus auxiliary terms (exp, log, 1/x) for non-polynomial components.
    All basis columns are standardized to unit variance for well-conditioned fitting.
    
    params[0:10] are coefficients for 10 standardized basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    x_safe = np.clip(x, 1e-10, None)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev polynomials
    a, b = 0.1, 6.0
    t = (2.0 * x_safe - (a + b)) / (b - a)
    t = np.clip(t, -1.0, 1.0)
    
    # Chebyshev polynomials T_0 through T_6
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    T6 = 32.0 * t**6 - 48.0 * t**4 + 18.0 * t**2 - 1.0
    
    # Auxiliary terms
    exp_term = np.exp(np.clip(x_safe, -50, 50))
    log_term = np.log(x_safe)
    inv_term = 1.0 / x_safe
    
    # Stack into matrix: shape (n, 10)
    basis = np.column_stack([T0, T1, T2, T3, T4, T5, T6, exp_term, log_term, inv_term])
    
    # Standardize each column to unit variance (zero mean, unit std)
    means = basis.mean(axis=0)
    stds = basis.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    basis_std = (basis - means) / stds
    
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Predict: (n, 10) @ (10,) -> (n,)
    pred = basis_std @ params
    
    return pred
# EVOLVE-BLOCK-END
