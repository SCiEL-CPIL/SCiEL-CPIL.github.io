import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization
    of sphere 1D minimization.
    
    We sample a single realization from the Gaussian posterior at each
    candidate point: x ~ N(mu, sigma^2), and return -x as the score.
    Higher score means lower sampled value, so we pick the candidate
    with the lowest sampled function value.
    
    This naturally balances exploration (high sigma regions get sampled
    values that can be very low) and exploitation (low mu regions tend
    to give low samples).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: pure exploitation
        return -mu.copy()
    
    # Handle near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Sample from the posterior: f(x) ~ N(mu(x), sigma(x)^2)
    # Use a single random standard normal vector for efficiency
    eps = np.random.standard_normal(len(mu))
    sampled_values = mu + sigma_safe * eps
    
    # Return negative sampled values as scores (higher = better for selection)
    scores = -sampled_values
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
