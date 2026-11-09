import numpy as np
import time

# Global start time to track elapsed time for the deadline guard
_start_time = time.time()
# The evaluator kills the program after 330s wall-clock.
# We set a deadline margin to ensure we return before the kill switch.
_DEADLINE_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for minimization.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > 330 - _DEADLINE_MARGIN:
        # Fallback: Return a simple deterministic score to ensure termination.
        # Greedy approach: pick the point with the lowest mean.
        # Score = -mu, so max score corresponds to min mu.
        return -np.asarray(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and safe for sampling
    # If sigma is 0, the sample is just mu.
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling:
    # Draw samples from the posterior distribution N(mu, sigma^2)
    # We are minimizing f. We want to find the x that minimizes f(x).
    # Sample f(x) = mu(x) + sigma(x) * z, where z ~ N(0, 1)
    # The evaluator selects the candidate with the HIGHEST score.
    # We want the evaluator to select the candidate with the LOWEST sampled f(x).
    # Therefore, we return score = -f_sample.
    
    z = np.random.standard_normal(len(mu))
    f_samples = mu + sigma_safe * z
    
    # Score is negative of the sample
    scores = -f_samples
    
    # Ensure all scores are finite
    # If any non-finite values arise (unlikely with standard_normal and finite mu/sigma), replace them
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
