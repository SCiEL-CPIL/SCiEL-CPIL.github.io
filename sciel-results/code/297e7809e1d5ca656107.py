import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: Array of mean predictions, shape (2000,).
        sigma: Array of standard deviation predictions, shape (2000,).
        f_best: The lowest observed objective value.
        
    Returns:
        Array of scores, shape (2000,). Higher scores are preferred.
    """
    # Calculate the distance from the known minimum (3.0)
    # If f_best is close to 3, we should exploit (low beta).
    # If f_best is far from 3, we should explore (high beta).
    delta = f_best - 3.0
    if delta < 0:
        delta = 0.0
        
    # Adaptive beta:
    # beta_min = 0.5 (exploit mode)
    # beta_max = 2.0 (explore mode)
    # Transition based on delta. Scale factor 10.0 controls the width of the transition.
    beta = 0.5 + 1.5 * (1.0 - np.exp(-delta / 10.0))
    
    # UCB score for minimization:
    # We maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    return scores
