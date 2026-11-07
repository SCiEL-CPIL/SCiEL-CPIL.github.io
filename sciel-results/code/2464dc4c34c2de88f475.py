import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0
_TIME_MARGIN = 10.0 # Safety margin in seconds

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: Mean of the GP posterior at candidate points (shape: (N,))
        sigma: Standard deviation of the GP posterior at candidate points (shape: (N,))
        f_best: The best observed value so far (scalar)
        
    Returns:
        scores: Scores for each candidate point, higher means evaluate next (shape: (N,))
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - _TIME_MARGIN:
        # Fallback to a fast, deterministic method (e.g., simple EI or UCB) if time is up
        # Using UCB with beta=2.0 as a safe fallback
        return -mu + 2.0 * sigma

    # Thompson Sampling: Sample from the posterior distribution
    # We want to select the point with the lowest function value.
    # The evaluator picks the point with the HIGHEST score.
    # So we return the negative of the sample.
    samples = np.random.normal(mu, sigma)
    
    # Ensure scores are finite
    scores = -samples
    scores[~np.isfinite(scores)] = -np.inf
    
    return scores
