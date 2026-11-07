"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
TIME_LIMIT = 290  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Minimum observed value
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # If we are running out of time, fall back to a simple deterministic score
        # to ensure we return something valid quickly.
        # Since we are in the acquisition function, we can't do much heavy work.
        # Returning negative mu is a reasonable exploitation fallback.
        return -mu

    # Draw random samples from the posterior
    # epsilon ~ N(0, 1)
    eps = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled function values
    # f_sample = mu + sigma * eps
    # We want to MINIMIZE f_sample.
    # The evaluator picks the MAX score.
    # So we return -f_sample.
    scores = -(mu + sigma * eps)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
