"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Instead of computing a deterministic utility (like EI or UCB), we draw
    a single random sample from the posterior predictive distribution
    f ~ N(mu, sigma^2) and return these samples as scores. The candidate
    with the highest sampled value is selected next.
    
    This naturally balances exploration and exploitation:
    - High sigma regions get random upward (or downward) kicks, encouraging exploration
    - Low sigma regions rely more on mu, encouraging exploitation of good areas
    
    For minimization, we want to find the lowest f(x), so we return the
    negative of the sample (higher score = lower predicted value = better).
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from the posterior
    # f_sample ~ N(mu, sigma^2)
    # For minimization, we want low values, so we negate to get "higher is better"
    # score = -f_sample = -(mu + sigma * z) where z ~ N(0,1)
    z = np.random.standard_normal(len(mu))
    f_sample = mu + sigma_safe * z
    
    # For minimization: higher score means lower predicted value
    scores = -f_sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (should be very fast, but guard just in case)
    elapsed = time.time() - start_time
    if elapsed > 299.0:
        # Fallback: return deterministic EI if somehow we're running out of time
        z_ei = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
        scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
