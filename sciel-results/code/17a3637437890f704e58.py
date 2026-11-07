import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB for minimization
    # Score = -mu + beta * sigma
    # Higher score is better.
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, None)
    
    # Choose beta
    # A moderate beta encourages exploration.
    # Since we don't know the step, we use a fixed value.
    # 2.0 to 4.0 is typical.
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
