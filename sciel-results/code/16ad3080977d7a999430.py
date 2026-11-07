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
    deadline = start_time + 250  # Safety margin before 300s limit
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check time before returning
    if time.time() > deadline:
        return ei
    
    return ei
# EVOLVE-BLOCK-END
