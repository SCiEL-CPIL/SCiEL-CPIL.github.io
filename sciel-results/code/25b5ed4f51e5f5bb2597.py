"""Symbolic regression with 1 input(s) using a dense Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of 10 Gaussian RBFs.
    
    The basis consists of 10 Gaussian kernels with centers evenly spaced across
    the input domain [0.1, 6]. This provides high local flexibility to capture
    sharp features, bumps, and non-polynomial structures that global bases
    like Chebyshev polynomials might smooth over.
    
    The linear-in-parameters structure ensures the optimization problem is
    convex with respect to the parameters, facilitating convergence with BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Define RBF centers evenly spaced in [0.1, 6.0]
    # 10 centers to match the 10 parameters
    centers = np.linspace(0.1, 6.0, 10)
    
    # Bandwidth sigma
    # A smaller sigma allows for sharper features but increases condition number.
    # 0.4 provides a good balance for the domain size ~5.9
    sigma = 0.4
    sigma_sq = sigma * sigma
    
    # Compute RBF basis
    # x is (N,), centers is (10,)
    # diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Basis terms: exp(-diff^2 / (2 * sigma^2))
    basis = np.exp(-(diff ** 2) / (2.0 * sigma_sq))
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
