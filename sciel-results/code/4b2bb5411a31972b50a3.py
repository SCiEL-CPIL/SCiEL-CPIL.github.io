import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    mu: array-like, shape (n_candidates,)
        Mean of the Gaussian Process posterior.
    sigma: array-like, shape (n_candidates,)
        Standard deviation of the Gaussian Process posterior.
    f_best: float
        The best (minimum) value observed so far.
        
    Returns:
    scores: array-like, shape (n_candidates,)
        UCB scores. Higher is better.
    """
    # Ensure sigma is positive to avoid issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to find low values.
    # A good point has a low mean (mu) and high uncertainty (sigma).
    # Score = -mu + kappa * sigma
    # Higher score means more likely to be a good minimum.
    
    kappa = 1.0
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
