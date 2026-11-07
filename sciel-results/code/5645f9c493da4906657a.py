import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # We maximize the potential for finding a low value, penalized by cost.
    # Score = (beta * sigma - mu) / cost
    
    # Hyperparameter for exploration-exploitation trade-off
    beta = 1.0
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB score for minimization: we want mu to be low and sigma to be high
    ucb_scores = beta * sigma - mu
    
    # Normalize by cost to prefer cheaper evaluations
    # Cost is at least 1.0
    scores = ucb_scores / (cost + 1e-9)
    
    return scores
