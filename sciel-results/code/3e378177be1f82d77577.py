"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Linear + Sum of 4 Gaussians.
    params[0]: intercept
    params[1]: slope
    params[2:6]: amplitudes for 4 Gaussians
    params[6:10]: log-widths for 4 Gaussians (exp applied to ensure positivity)
    """
    # Safety check for non-finite inputs
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(params)):
        return np.zeros_like(x)

    a0 = params[0]
    a1 = params[1]
    
    # Fixed centers for the 4 Gaussian components
    # Spaced across [-2, 2]
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    
    # Extract parameters for Gaussians
    amps = params[2:6]
    log_widths = params[6:10]
    
    # Clip log_widths to prevent extreme values in exp
    # exp(-10) is ~4.5e-5, exp(10) is ~22000
    # Widths should be reasonable, e.g., 0.01 to 100
    # log(0.01) ~ -4.6, log(100) ~ 4.6
    log_widths_clipped = np.clip(log_widths, -5.0, 5.0)
    widths = np.exp(log_widths_clipped)
    
    # Compute Gaussian terms
    # x shape: (N,)
    # centers shape: (4,)
    # We want output shape (N,)
    
    x_col = x[:, np.newaxis]  # (N, 1)
    c_row = centers[np.newaxis, :]  # (1, 4)
    
    # Squared distances
    dist_sq = (x_col - c_row) ** 2  # (N, 4)
    
    # Denominator: 2 * width^2
    # widths shape: (4,)
    denom = 2.0 * (widths ** 2)  # (4,)
    
    # Exponents
    exponents = -dist_sq / denom  # (N, 4)
    
    # Clip exponents to avoid underflow (which is fine) or potential numerical issues
    # np.exp handles underflow to 0.0 safely
    gaussians = amps[np.newaxis, :] * np.exp(exponents)  # (N, 4)
    
    # Sum over the 4 Gaussian components
    gaussian_sum = np.sum(gaussians, axis=1)
    
    # Combine with linear part
    result = a0 + a1 * x + gaussian_sum
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to linear if something went wrong
        result = np.where(np.isfinite(result), result, a0 + a1 * x)
        
    return result
# EVOLVE-BLOCK-END
