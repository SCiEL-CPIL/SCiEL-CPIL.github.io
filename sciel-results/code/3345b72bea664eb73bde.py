import numpy as np
import time

# Global state to track time for the current optimization run
_start_time = None
_time_limit = 290.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize time tracker if this is the first call in a run
    if _start_time is None:
        _start_time = time.time()
    else:
        # If we've exceeded the safe time limit, fall back to a simple greedy strategy
        # to ensure we return quickly and don't get killed
        if time.time() - _start_time > _time_limit:
            # Reset start time? No, we are likely in the same run.
            # Just return a fast score.
            # Greedy: pick lowest mu.
            return -mu

    # Thompson Sampling
    # Sample from N(mu, sigma^2)
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Generate samples
    # Using standard normal and scaling
    samples = mu + sigma * np.random.standard_normal(mu.shape)
    
    # Cost-aware score: Lower sample is better, lower cost is better.
    # Score = -sample / cost
    # This encourages low function values and low costs.
    
    # Handle potential division issues, though cost >= 1
    scores = -samples / cost
    
    return scores
