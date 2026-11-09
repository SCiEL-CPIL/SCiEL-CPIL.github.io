"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses a pure Expected Improvement (EI) acquisition with careful numerical handling.
    EI is theoretically optimal for maximizing expected gain and avoids suboptimal
    exploration/exploitation balance from adaptive heuristics.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z) with numerical stability
    # For very negative z, Phi(z) ~ 0, so EI ~ 0
    # For very positive z, Phi(z) ~ 1, so EI ~ f_best - mu
    
    # Use scipy's norm.cdf and norm.pdf for stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Compute EI
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Handle numerical edge cases
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure non-negative EI (numerical errors can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
