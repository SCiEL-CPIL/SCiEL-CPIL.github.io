import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) strategy.
    For minimization, we maximize the score: -mu + beta * sigma.
    This encourages exploration of regions with high uncertainty.
    """
    beta = 2.0
    
    # Avoid division by zero or numerical issues, though UCB doesn't divide by sigma.
    # However, we ensure sigma is non-negative.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB score for minimization:
    # We want to find points where the true function value is likely low.
    # The lower bound of the confidence interval is mu - beta * sigma.
    # Minimizing this lower bound is equivalent to maximizing -mu + beta * sigma.
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
