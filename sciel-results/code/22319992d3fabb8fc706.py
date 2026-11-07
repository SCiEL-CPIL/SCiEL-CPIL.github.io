import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f, so we look for low mu and high sigma.
    # LCB = mu - beta * sigma. We want to minimize LCB.
    # Since we return scores where higher is better, we return -LCB / cost.
    
    beta = 2.5
    cost_safe = np.maximum(cost, 1e-9)
    
    # Handle potential NaNs in mu or sigma
    mu_safe = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    lcb = mu_safe - beta * sigma_safe
    scores = -lcb / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
