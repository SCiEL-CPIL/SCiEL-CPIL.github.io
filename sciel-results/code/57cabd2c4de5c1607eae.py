"""Symbolic regression with 1 input(s) using Hybrid Chebyshev and RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a hybrid basis:
    5 Chebyshev polynomials for global structure and 5 Gaussian RBFs for local flexibility.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # 1. Chebyshev Part (5 terms: T_0 to T_4)
    # Map x from [0.1, 6] to t in [-1, 1]
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    cheb_basis = np.zeros((n_points, 5), dtype=np.float64)
    cheb_basis[:, 0] = 1.0
    cheb_basis[:, 1] = t
    for k in range(2, 5):
        cheb_basis[:, k] = 2.0 * t * cheb_basis[:, k-1] - cheb_basis[:, k-2]
        
    # 2. RBF Part (5 Gaussian kernels)
    # Centers evenly spaced in [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 5)
    sigma = 1.0 # Bandwidth
    
    # Compute RBFs: exp(-(x - c)^2 / (2 * sigma^2))
    # x is (N, 1), centers is (1, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_basis = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Combine bases
    # Total 10 parameters
    basis = np.hstack((cheb_basis, rbf_basis))
    
    # Linear combination
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
