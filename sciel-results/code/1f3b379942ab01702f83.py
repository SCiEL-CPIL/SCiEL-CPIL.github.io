import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma
    """
    # Beta controls the exploration/exploitation trade-off
    # A value of 1.0 is standard for 1-sigma bounds
    # Given the multimodal nature of Goldstein-Price, a slightly higher beta 
    # might be beneficial, but 1.0 is a safe baseline.
    beta = 1.0
    
    # Calculate UCB score
    # We want to minimize f(x).
    # The lower bound of the prediction is mu - beta * sigma.
    # We want to find x that minimizes this lower bound.
    # So we maximize the negative of it: -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
