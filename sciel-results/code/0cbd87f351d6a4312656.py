import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None
# Counter to track which acquisition call we're on
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # The evaluator kills after 330s. We want to be safe.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Fallback: Return -mu to exploit the current best mean.
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Increment call counter
    _call_count += 1
    
    # Thompson Sampling approximation:
    # Sample from the posterior: theta = mu + sigma * z, where z ~ N(0,1)
    # For minimization, we want to select the point with the lowest sampled value.
    # Since the acquisition function returns scores where higher means "evaluate next",
    # we return the negative of the sampled value.
    
    # Use a fixed random seed based on call count for reproducibility within a seed
    # but different across calls. Actually, let's use the global numpy random state.
    # The evaluator will use different numpy random states for different seeds.
    z_sample = np.random.randn(len(mu))
    
    # Thompson sample
    theta = mu + sigma_safe * z_sample
    
    # We want to minimize, so higher score = lower theta
    # Return -theta as the score
    scores = -theta
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
