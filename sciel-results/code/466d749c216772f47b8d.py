import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize: beta * sigma - mu
    # This encourages exploration (high sigma) and exploitation (low mu)
    
    # Choose a beta parameter. 
    # A value around 1.0-2.0 is typical for standardized objectives.
    # Since we don't have dynamic tuning, we use a fixed value.
    # 1.5 is a reasonable trade-off for early-stage optimization.
    beta = 1.5
    
    # Calculate UCB score
    ucb = beta * sigma - mu
    
    # Cost-aware score: maximize UCB per unit cost
    score = ucb / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
