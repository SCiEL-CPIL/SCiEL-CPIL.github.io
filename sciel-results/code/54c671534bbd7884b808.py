"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    Samples from the posterior predictive at each candidate point,
    then selects the candidate with the lowest sampled value.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: pure exploitation
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) at each point
    # The score is the negative of the sampled value, so higher score = lower predicted value
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(mu.shape)
    
    # For minimization, we want the lowest sampled value to have the highest score
    # Score = -sampled_value
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
