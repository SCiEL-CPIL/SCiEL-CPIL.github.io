import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 320s to leave 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling with deadline guard.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points.
    sigma (np.ndarray): Posterior standard deviation at candidate points.
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point. Higher score is better.
    """
    # Deadline guard: if we're too close to the limit, use a fast deterministic fallback
    if time.time() > _DEADLINE - 5.0:
        # Fallback: use LCB with small kappa for speed
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.1 * sigma_safe)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Use Thompson Sampling: draw from N(mu, sigma^2)
    # For minimization, we want to find points where f(x) is likely small
    # Sample from the posterior predictive
    samples = mu + sigma * np.random.standard_normal(mu.shape)
    
    # Return negative samples so higher score = better (lower f)
    return -samples
