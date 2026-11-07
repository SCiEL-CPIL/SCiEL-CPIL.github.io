"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a Radial Basis Function (RBF) approach with Gaussian kernels.
    # This provides a local basis, distinct from the global polynomial basis.
    # We fix the centers and widths, and optimize only the weights (params).
    # This makes the problem linear in params, ensuring robust convergence with BFGS.
    
    # Input x is in [0.1, 6]
    # Define 10 centers uniformly spaced in this range
    x_min = 0.1
    x_max = 6.0
    n_basis = 10
    
    centers = np.linspace(x_min, x_max, n_basis)
    
    # Width of the Gaussians. 
    # Distance between centers is (6.0 - 0.1) / 9 ≈ 0.655
    # A sigma of 0.5 provides good overlap but distinct peaks.
    sigma = 0.5
    
    # Compute the basis matrix
    # x shape: (N,)
    # centers shape: (10,)
    # We want (N, 10)
    
    # x[:, None] - centers[None, :] gives (N, 10)
    diff = x[:, None] - centers[None, :]
    
    # Gaussian: exp(-0.5 * (diff/sigma)^2)
    # To avoid underflow for large diffs (though max diff is ~6), this is safe.
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction is basis @ params
    return basis @ params
# EVOLVE-BLOCK-END
