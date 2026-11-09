import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Pure GP-Lower Confidence Bound (GP-LCB) acquisition for minimization.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # GP-LCB for minimization: score = -mu - kappa * sigma
    # Higher scores are preferred (we want to minimize mu + kappa * sigma)
    # So we negate: score = -(mu + kappa * sigma) = -mu - kappa * sigma
    kappa = 2.0
    scores = -mu - kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
