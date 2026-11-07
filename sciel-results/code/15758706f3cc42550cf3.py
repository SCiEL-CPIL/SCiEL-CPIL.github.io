"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Strategy: Fixed Gaussian RBF Basis
    # This approach uses 10 Gaussian kernels with fixed centers and widths.
    # It is a linear model in the parameters, suitable for BFGS.
    # It can capture non-polynomial smooth functions better than a fixed-degree polynomial.
    
    n_kernels = 10
    # Centers evenly spaced across the domain [-2, 2]
    centers = np.linspace(-2.0, 2.0, n_kernels)
    
    # Width parameter. 
    # A width of 1.0 provides good overlap and smoothness.
    sigma = 1.0
    
    # Flatten x to ensure 1D
    x_flat = x.flatten()
    
    # Compute differences: (N, 1) - (1, K) -> (N, K)
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian basis functions
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of basis functions
    # params is (K,), basis is (N, K)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
