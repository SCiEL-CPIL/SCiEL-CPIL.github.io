import numpy as np
from scipy.stats import norm
import time

# Global start time to track elapsed time for the entire evaluation session
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes the Thompson Sampling acquisition function.
    
    This function implements Thompson Sampling by drawing samples from the
    posterior distribution of the objective function at each candidate point.
    Since the evaluator selects the candidate with the highest score, and we
    are minimizing the objective, we return the negative of the sampled values.
    This effectively selects the candidate with the lowest sampled objective value,
    balancing exploration and exploitation through the natural variance of the
    posterior.
    
    Parameters:
    mu: array of shape (n_candidates,) - posterior mean
    sigma: array of shape (n_candidates,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (n_candidates,) - Negative sampled values (higher is better)
    """
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback to a simple deterministic score if we are out of time
        # This is unlikely to be hit as the operation is O(N)
        pass

    # Ensure sigma is positive to avoid division by zero or invalid sampling
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw samples from the standard normal distribution
    z = np.random.standard_normal(mu.shape)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    f_samples = mu + sigma * z
    
    # We want to minimize f, but the caller picks the max score.
    # Therefore, we return -f_samples.
    scores = -f_samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
