"""Symbolic regression with a robust hybrid model: rational + exponential + log + polynomial."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with fixed centers and width.
    
    Structure:
    y = sum_{i=0}^{8} w_i * exp(-(u - c_i)^2 / (2 * sigma^2)) + b
    
    Where:
    u = (x - 3.05) / 2.95 (standardized to approx [-1, 1])
    c_i are fixed centers evenly spaced in [-1, 1]
    sigma is a fixed width parameter
    
    Parameters (10 total):
    p[0..8]: Weights for the 9 RBF kernels
    p[9]: Bias term
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Ensure p has length 10
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Standardize x: map [0.1, 6] to approx [-1, 1]
    # Center: (0.1 + 6) / 2 = 3.05, Scale: (6 - 0.1) / 2 = 2.95
    x_center = 3.05
    x_scale = 2.95
    u = (x - x_center) / x_scale
    
    # Define fixed RBF centers in the standardized domain [-1, 1]
    # 9 centers: -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0
    centers = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Fixed width (sigma)
    # A smaller sigma makes the basis functions more local.
    # With 9 centers over [-1, 1], spacing is 0.25.
    # sigma=0.3 provides good overlap for smooth functions.
    sigma = 0.3
    sigma_sq_2 = 2.0 * sigma * sigma
    
    # Calculate RBF basis values
    # u shape: (N, 1) or (N,)
    # centers shape: (9,)
    # diff shape: (N, 9)
    diff = u[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exp(-(diff^2) / (2*sigma^2))
    # Clip the exponent to avoid underflow to 0 (which is fine) or overflow (not possible here as exponent is negative)
    # Actually, for large negative exponents, exp is 0. For small, it's ~1.
    # No overflow risk since exponent <= 0.
    basis = np.exp(-diff**2 / sigma_sq_2)
    
    # Combine with weights
    # weights shape: (9,)
    # result shape: (N,)
    result = basis @ p[:9] + p[9]
    
    # Final safety check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
