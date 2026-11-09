import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We want to maximize the score, which corresponds to the lower bound of the confidence interval
    # Lower bound = mu - beta * sigma
    # Score = - (mu - beta * sigma) = -mu + beta * sigma
    
    beta = 2.5
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
