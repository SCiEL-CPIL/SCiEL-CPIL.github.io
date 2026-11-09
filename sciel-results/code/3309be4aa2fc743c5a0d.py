import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    mu: array of predicted means (shape N,)
    sigma: array of predicted standard deviations (shape N,)
    f_best: best observed function value so far
    
    Returns:
    scores: array of scores (shape N,), higher is better
    """
    # UCB balances exploitation (low mu) and exploration (high sigma)
    # For minimization, we maximize -mu + beta * sigma
    # beta controls the exploration-exploitation trade-off
    # A value around 2.0 - 3.0 is typical for multimodal functions
    beta = 2.5
    
    # Ensure sigma is non-negative to avoid numerical issues, though GP should return positive
    sigma = np.maximum(sigma, 0.0)
    
    scores = -mu + beta * sigma
    
    return scores
