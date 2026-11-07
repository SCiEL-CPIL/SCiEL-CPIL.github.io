"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based symbolic regression.
    
    Strategy:
    Instead of a global polynomial basis which requires the optimizer to balance
    conflicting high-order terms, we use a set of fixed Radial Basis Functions (Gaussians).
    The centers are fixed on a grid across the domain [0.1, 6.0].
    The widths are fixed to provide local coverage.
    The 10 parameters are the weights for these 10 RBFs.
    
    This transforms the problem into a linear combination of local "bumps".
    BFGS can easily adjust the height of each bump to fit the target function
    without the ill-conditioning issues of high-degree polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Define 10 fixed centers across the domain [0.1, 6.0]
    # We use linspace to ensure even coverage.
    # Using 10 centers allows us to use all 10 parameters as weights.
    centers = np.linspace(0.1, 6.0, 10)
    
    # Define fixed widths.
    # The spacing between centers is (6.0 - 0.1) / 9 = 5.9 / 9 ≈ 0.655.
    # A width of ~0.3-0.5 ensures overlap for smoothness but distinct peaks.
    # Let's use a width that is roughly half the spacing to allow good local fitting.
    width = 0.35
    
    # Compute RBF values: exp(-((x - center)^2) / (2 * width^2))
    # x shape: (N,)
    # centers shape: (10,)
    # We want an output of shape (N, 10)
    
    # Reshape for broadcasting
    x_col = x[:, np.newaxis]  # (N, 1)
    centers_row = centers[np.newaxis, :]  # (1, 10)
    
    # Compute squared distances
    dist_sq = (x_col - centers_row) ** 2
    
    # Compute Gaussian RBFs
    rbf_values = np.exp(-dist_sq / (2 * width**2))
    
    # Linear combination of RBFs using the 10 parameters
    # rbf_values: (N, 10)
    # params: (10,)
    # result: (N,)
    return rbf_values @ params
# EVOLVE-BLOCK-END
