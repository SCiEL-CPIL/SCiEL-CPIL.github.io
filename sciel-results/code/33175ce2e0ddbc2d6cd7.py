import numpy as np
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # The evaluator kills after 330s. We want to be safe.
    # This function is called every step. The heavy lifting is in the GP update (not here),
    # but we ensure this function itself is fast and safe.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Fallback: Return -mu to exploit the current best mean.
        # This is a safe, deterministic choice that ensures we don't timeout 
        # by doing complex sampling if we are running out of time.
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Thompson Sampling:
    # Sample a potential function value for each candidate from its posterior.
    # f_sampled = mu + sigma * z
    # We want to minimize f, so we maximize -f_sampled.
    
    # Generate standard normal random numbers
    z = np.random.standard_normal(mu.shape)
    
    # Compute sampled values
    sampled_values = mu + sigma_safe * z
    
    # The score should be higher for better (lower) function values.
    # So we negate the sampled values.
    scores = -sampled_values
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
