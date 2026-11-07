import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean of the GP posterior (2000,)
        sigma: Standard deviation of the GP posterior (2000,)
        f_best: Best observed value so far (scalar)
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        Scores (2000,) where higher is better.
    """
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: We want to maximize (-mu + kappa * sigma)
    # kappa controls exploration. 1.0 is a standard starting point.
    # Given the small budget, we might want slightly more exploration.
    kappa = 1.5
    
    ucb_scores = -mu + kappa * sigma_safe
    
    # Normalize by cost
    scores = ucb_scores / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
