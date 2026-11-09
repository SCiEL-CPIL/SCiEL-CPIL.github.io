"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Draws a single sample from the posterior predictive distribution at each
    candidate point and returns the negative of that sample as the score.
    Higher scores correspond to lower predicted objective values, which
    inherently balances exploration and exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best observed objective value so far (lowest value seen).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Track time to ensure we finish within budget
    start_time = time.time()
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single standard normal sample for Thompson Sampling
    # We use a fixed seed based on the current state to make it deterministic per call
    # but stochastic across calls. However, since we need reproducibility and the
    # evaluator calls this multiple times, we should use a global or thread-local
    # random state. For simplicity and to avoid issues, we'll use numpy's default
    # random state which is seeded by the system.
    
    # Draw z ~ N(0, 1) for each candidate point
    z = np.random.standard_normal(len(mu))
    
    # Thompson sample: f_sample = mu - sigma * z
    # We want to minimize f, so we score by -f_sample
    # Lower f_sample is better, so higher -f_sample is better
    f_sample = mu - sigma_safe * z
    scores = -f_sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check time budget - if we're running out of time, return immediately
    elapsed = time.time() - start_time
    # The computation above is very fast (microseconds), so this check is just a safety net
    
    return scores
# EVOLVE-BLOCK-END
