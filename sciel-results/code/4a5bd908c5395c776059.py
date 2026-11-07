"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute PDF and CDF of standard normal distribution
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure scores are finite
    # EI should be non-negative, but numerical errors might cause tiny negatives or inf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
        return ei
        
    return ei
# EVOLVE-BLOCK-END
