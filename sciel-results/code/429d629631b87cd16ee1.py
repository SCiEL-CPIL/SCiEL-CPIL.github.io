"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Handle edge cases where sigma might be zero or extremely small
    # to avoid division by zero in the z-score calculation.
    # np.where is vectorized and fast.
    sigma_safe = np.where(sigma > 1e-8, sigma, 1.0)
    
    # Calculate z-score for minimization:
    # Improvement is defined as f_best - f(x).
    # We want E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then (f_best - f(x)) ~ N(f_best - mu, sigma^2)
    # Let Z = (f_best - mu) / sigma.
    # EI = sigma * (Z * Phi(Z) + phi(Z))
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Using scipy.stats.norm for numerical stability of pdf and cdf
    ei = sigma_safe * (z * norm.cdf(z) + norm.pdf(z))
    
    # Ensure non-negative EI (due to floating point errors)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
