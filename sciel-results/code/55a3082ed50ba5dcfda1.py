"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 330s limit, use 300s as safety margin
    if time.time() - _bo_start_time > 300:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu
    
    # Thompson Sampling acquisition function
    # Draw a single random sample from the posterior distribution
    # theta = mu + sigma * Z, where Z ~ N(0, 1)
    # Score candidates based on -theta (minimize theta)
    
    # Generate standard normal random numbers
    Z = np.random.randn(mu.shape[0])
    
    # Compute Thompson sample
    theta = mu + sigma * Z
    
    # Score is negative of theta (we want to minimize theta, so maximize -theta)
    scores = -theta
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
