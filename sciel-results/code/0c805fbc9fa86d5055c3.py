import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB (Upper Confidence Bound) for minimization
    # We want to maximize: -mu + kappa * sigma
    # Then normalize by cost.
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration constant
    # With ~25 evaluations in 2D, we need good exploration.
    # A value around 2-3 is typical.
    kappa = 2.5
    
    # Calculate UCB score
    # Note: We are minimizing, so we look for low mu.
    # -mu rewards low mu.
    # sigma rewards uncertainty.
    ucb = -mu + kappa * sigma
    
    # Cost-aware score
    # Avoid division by zero, though cost >= 1
    cost_safe = np.maximum(cost, 1e-9)
    
    score = ucb / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
