import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want low mu and high sigma
    # Optimistic value: mu - beta * sigma
    # We want to maximize the "potential" which is - (mu - beta * sigma) = beta * sigma - mu
    beta = 2.0  # Exploration-exploitation trade-off
    ucb = beta * sigma_safe - mu
    
    # Cost-aware score
    score = ucb / cost
    
    # Ensure finite
    return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
