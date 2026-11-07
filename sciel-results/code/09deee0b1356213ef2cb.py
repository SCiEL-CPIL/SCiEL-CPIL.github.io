# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better)
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Ensure inputs are proper float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values in mu and sigma with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values that can cause numerical issues
    z = np.clip(z, -100, 100)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check time before returning
    if time.time() > deadline:
        return ei
    
    return ei
# EVOLVE-BLOCK-END
