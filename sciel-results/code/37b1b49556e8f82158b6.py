import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    # Deadline guard
    # The computation is fast, but we check just in case.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Epsilon to prevent issues with zero/negative sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Structural change: Use UCB (Upper Confidence Bound) for minimization
    # For minimization, we use LCB = mu - kappa * sigma
    # We want high scores for points with low LCB, so score = -LCB = -mu + kappa * sigma
    # This is a different optimization principle than EI:
    # EI maximizes expected improvement, UCB/LCB maximizes the optimistic (lower) bound
    kappa = 1.0
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
         # If somehow slow, return a fallback
         # But this shouldn't happen
         pass
         
    return scores
