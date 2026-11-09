"""Symbolic regression with adaptive Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using 5 Gaussian Radial Basis Functions.
    
    Architecture:
    - 5 fixed centers: -3.0, -1.5, 0.0, 1.5, 3.0
    - 5 learnable widths (parameterized as exp(p_i) for stability)
    - 5 learnable amplitudes
    
    Total parameters: 10 (5 widths + 5 amplitudes)
    
    This localized basis allows the model to capture non-polynomial features
    and local variations in the target function, unlike global polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Fixed centers spread across the domain [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract parameters
    log_widths = params[0:5]
    amplitudes = params[5:10]
    
    # Compute widths from log-widths to ensure positivity
    # Clip to prevent extreme values that might cause numerical issues
    # exp(1) ~ 2.7, which is a reasonable default width
    widths = np.exp(log_widths)
    widths = np.clip(widths, 0.1, 5.0)
    
    # Compute RBFs
    # x shape: (n, 1) or (n,)
    # centers shape: (5,)
    # We need (n, 5) matrix of distances
    x_col = x.reshape(-1, 1)
    dists = x_col - centers.reshape(1, -1)
    
    # Gaussian RBF: exp(-d^2 / (2 * sigma^2))
    # Use (dists / widths)^2 to avoid division by zero (widths clipped > 0)
    scaled_dists = dists / widths
    rbf_values = np.exp(-0.5 * scaled_dists ** 2)
    
    # Combine with amplitudes
    # rbf_values shape: (n, 5)
    # amplitudes shape: (5,)
    result = rbf_values @ amplitudes
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
