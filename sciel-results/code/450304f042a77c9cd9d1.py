"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Expected Improvement (EI).
    
    Args:
        mu: array of shape (2000,) with posterior means.
        sigma: array of shape (2000,) with posterior standard deviations.
        f_best: scalar, the lowest observed value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    # For z > 8: Phi(z) ~ 1, phi(z) ~ 0
    # For z < -8: Phi(z) ~ 0, phi(z) ~ 0
    z_clipped = np.clip(z, -8.0, 8.0)
    
    # Compute EI using clipped z for stability
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # For very large positive z, EI should be approximately f_best - mu
    # For very negative z, EI should be approximately 0
    # The clipping already handles this well
    
    # Ensure finite output
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
