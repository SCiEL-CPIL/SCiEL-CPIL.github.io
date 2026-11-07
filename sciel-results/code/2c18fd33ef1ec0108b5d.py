import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling for minimization.
    
    mu: (N,) array of posterior means
    sigma: (N,) array of posterior std devs
    f_best: float, best observed value so far
    Returns: (N,) array of scores (higher = better to evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe default: pure exploitation (negative mean)
        mu_arr = np.asarray(mu)
        scores = -mu_arr
        scores = np.nan_to_num(scores, nan=-1e10, posinf=-1e10, neginf=-1e10)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=0.0)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Thompson Sampling: sample from the posterior
    # For minimization, we want to sample low values
    # Sample: theta = mu + sigma * z, where z ~ N(0, 1)
    # We want to maximize the score, so we return -theta
    # This effectively samples from the posterior and picks the one with lowest value
    
    # Generate random samples
    z = np.random.standard_normal(mu.shape)
    
    # Compute sampled values
    samples = mu + sigma * z
    
    # For minimization, higher score = lower sampled value
    # So score = -samples
    scores = -samples
    
    # Handle potential numerical issues
    scores = np.nan_to_num(scores, nan=-1e10, posinf=-1e10, neginf=-1e10)
    
    return scores
