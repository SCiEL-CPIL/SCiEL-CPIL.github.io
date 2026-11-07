"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 10 localized Gaussian RBFs + 10 global basis functions.
    
    The model is:
      f(x) = sum_{i=0}^{9} p[i] * phi_i(x)
    
    where phi_i(x) = exp(-((x - c_i)^2) / (2 * w_i^2)) + g_i(x)
    
    The 10 parameters p[i] weight the combined basis functions.
    Centers are evenly spaced in [-3, 3].
    Widths vary for multi-scale approximation.
    Global basis: constant, x, x^2, x^3, sin(x), cos(x), sin(2x), cos(2x), exp(-x^2/2), tanh(x)
    
    This is linear in parameters, so BFGS converges to the global least-squares minimum.
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size != 10:
        if p.size >= 10:
            p = p[:10]
        else:
            p = np.pad(p, (0, 10 - p.size))
    
    # Deadline guard
    if time.time() - start_time > 145.0:
        return np.zeros(n, dtype=np.float64)
    
    # Fixed centers: 10 points evenly spaced in [-3, 3]
    centers = -3.0 + 6.0 * np.arange(10) / 9.0  # shape (10,)
    
    # Fixed widths: varying for multi-scale approximation
    widths = np.array([0.5, 0.8, 1.1, 0.5, 0.8, 1.1, 0.5, 0.8, 1.1, 0.5])
    
    # Compute RBF values: shape (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # shape (n, 10)
    w2 = 2.0 * widths ** 2  # shape (10,)
    rbf_values = np.exp(-(diff ** 2) / w2[np.newaxis, :])  # shape (n, 10)
    
    # Compute global basis functions: shape (n, 10)
    b = np.ones((n, 10), dtype=np.float64)
    b[:, 0] = 1.0
    b[:, 1] = x
    b[:, 2] = x ** 2
    b[:, 3] = x ** 3
    b[:, 4] = np.sin(x)
    b[:, 5] = np.cos(x)
    b[:, 6] = np.sin(2.0 * x)
    b[:, 7] = np.cos(2.0 * x)
    b[:, 8] = np.exp(-0.5 * x ** 2)
    b[:, 9] = np.tanh(x)
    
    # Combine: each parameter multiplies (RBF_i + global_basis_i)
    # This gives a richer basis that captures both local and global structure
    combined_basis = rbf_values + b  # shape (n, 10)
    
    # Linear combination
    result = combined_basis @ p  # shape (n,)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
