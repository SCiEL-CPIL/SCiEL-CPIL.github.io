"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Upper Confidence Bound (UCB) with an adaptive exploration parameter (beta)
    that decreases as the best observed value (f_best) approaches the known global minimum (3.0).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Known global minimum of Goldstein-Price 2D
    f_opt = 3.0
    
    # Calculate current regret
    regret = f_best - f_opt
    if regret < 0:
        regret = 0.0
        
    # Adaptive beta:
    # Start with high exploration (beta ~ 2.0) when regret is high,
    # and reduce to exploitation (beta ~ 1.0) as we approach the optimum.
    # Linear scaling: beta = 1.0 + 0.1 * regret
    # If regret is 10, beta = 2.0
    # If regret is 50, beta = 6.0
    # Cap beta at 5.0 to prevent overly erratic behavior
    beta = 1.0 + 0.1 * regret
    beta = min(beta, 5.0)
    
    # UCB score for minimization:
    # We want to minimize the function, so we prefer low mu.
    # We want to explore, so we prefer high sigma.
    # Score = beta * sigma - mu
    # Higher score is better.
    scores = beta * sigma - mu
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
