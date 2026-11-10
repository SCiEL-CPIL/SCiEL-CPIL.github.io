"""Symbolic regression with a normalized Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a 10-term Chebyshev series on the scaled domain.
    x is scaled from [0.1, 6] to t in [-1, 1].
    Basis: T_0(t), T_1(t), ..., T_9(t)
    Each basis function is normalized by its RMS over the domain to balance
    gradient magnitudes across all 10 parameters, helping BFGS converge better.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(x) == 0:
        return np.array([])
    
    # Scale x from [0.1, 6] to t in [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    t = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T_0 through T_9
    n_terms = 10
    T = np.empty((n_terms, len(t)), dtype=float)
    T[0] = 1.0
    T[1] = t
    for n in range(2, n_terms):
        T[n] = 2.0 * t * T[n-1] - T[n-2]
    
    # Normalize each basis function by its RMS over the domain
    # Precompute normalization constants using a fine grid
    x_grid = np.linspace(0.1, 6.0, 1000)
    t_grid = (x_grid - 3.05) / 2.95
    
    basis_grid = np.empty((n_terms, len(t_grid)), dtype=float)
    basis_grid[0] = 1.0
    basis_grid[1] = t_grid
    for k in range(2, n_terms):
        basis_grid[k] = 2.0 * t_grid * basis_grid[k-1] - basis_grid[k-2]
    
    # Compute RMS for each basis function
    rms = np.sqrt(np.mean(basis_grid ** 2, axis=1))
    rms = np.where(rms < 1e-12, 1.0, rms)
    
    # Normalize the basis
    T_normalized = T / rms[np.newaxis, :]
    
    # Combine with params
    return np.sum(params[:, np.newaxis] * T_normalized, axis=0)
# EVOLVE-BLOCK-END
