import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure finite inputs
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in cost
    cost = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in z calculation if used, but LCB doesn't use z
    # LCB for minimization: minimize mu - kappa * sigma
    # We want to maximize: (kappa * sigma - mu) / cost
    
    kappa = 3.0  # Exploration-exploitation trade-off
    
    # Score is higher for lower mu and higher sigma
    score = (kappa * sigma - mu) / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
