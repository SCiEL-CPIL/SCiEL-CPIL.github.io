"""Hybrid approach: Polynomial trend + Localized Gaussian RBFs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a combination of global polynomial terms
    and localized Radial Basis Functions (Gaussian bumps).
    
    This hybrid basis is designed to capture both smooth global trends (via polynomials)
    and sharp localized features (via Gaussians).
    
    Parameters:
    params[0]: Offset (constant term)
    params[1]: Linear term coefficient
    params[2..9]: Weights for 8 Gaussian bumps centered at fixed locations.
    
    The centers are evenly spaced in [-3, 3].
    The width of the Gaussians is fixed to provide good local coverage.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.ravel()
    n = x.size
    
    if n == 0:
        return np.array([])
    
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite params
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Basis 0: Constant
    b0 = np.ones_like(x)
    
    # Basis 1: Linear
    b1 = x
    
    # Define 8 centers for Gaussian bumps
    # Evenly spaced in [-3, 3]
    # Using 8 centers to leave 2 params for global trend
    centers = np.linspace(-3.0, 3.0, 8)
    
    # Fixed width parameter for the Gaussians
    # lambda = 1.5 implies sigma ~ 0.57
    # This allows bumps to overlap slightly, ensuring smooth interpolation
    width = 1.5
    
    # Compute Gaussian basis functions
    # diff shape: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-width * diff ** 2)
    
    # Construct the full basis matrix
    # V shape: (N, 10)
    V = np.column_stack([
        b0,       # p0
        b1,       # p1
        gaussians # p2..p9
    ])
    
    result = V @ p
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
