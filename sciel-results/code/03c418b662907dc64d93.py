import numpy as np
import time

# Global timer for deadline guard
_MODULE_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    Score = -mu + kappa * sigma. Higher score is better.
    """
    # Deadline guard
    if time.time() - _MODULE_START_TIME > 300:
        # Fallback: Return 0s to avoid crashing, effectively random choice if forced
        return np.zeros_like(mu)

    # UCB parameter. 
    # A value of 2.0 encourages exploration. 
    # We can potentially adapt this, but a fixed value is robust for short runs.
    kappa = 2.0
    
    # Calculate UCB score
    # We want to minimize f, so we maximize -mu + kappa * sigma
    scores = -mu + kappa * sigma
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
