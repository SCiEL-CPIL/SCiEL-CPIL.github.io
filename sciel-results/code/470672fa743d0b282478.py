"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a hybrid basis combining local Gaussian RBFs and global trigonometric terms.
    
    This approach aims to capture both local sharp features (via RBFs) and global
    periodic or smooth trends (via sine/cosine terms), which may be present in
    the hidden target function.
    
    params: array of length 10.
    - params[0:6]: Weights for 6 Gaussian basis functions centered at -2, -1.2, -0.4, 0.4, 1.2, 2.
    - params[6:10]: Weights for 4 trigonometric basis functions: sin(x), cos(x), sin(2x), cos(2x).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Define 6 Gaussian centers
    # Spaced to cover [-2, 2] with good overlap
    centers = np.array([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0])
    
    # Fixed width for Gaussians
    sigma = 0.5
    
    # Compute Gaussian basis functions
    # Shape: (N, 6)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Compute trigonometric basis functions
    # Shape: (N, 4)
    trig_basis = np.empty((n, 4), dtype=np.float64)
    trig_basis[:, 0] = np.sin(x)
    trig_basis[:, 1] = np.cos(x)
    trig_basis[:, 2] = np.sin(2.0 * x)
    trig_basis[:, 3] = np.cos(2.0 * x)
    
    # Combine bases
    # Total shape: (N, 10)
    # First 6 columns are Gaussians, next 4 are trig
    # We can compute directly without concatenating large arrays if needed, but concat is fine for N=200
    basis = np.hstack((gaussians, trig_basis))
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
