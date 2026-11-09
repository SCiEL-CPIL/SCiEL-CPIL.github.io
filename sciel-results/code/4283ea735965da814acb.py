import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Avoid division by zero in cost
    cost = np.maximum(cost, 1e-6)
    
    # Avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term (UCB-like)
    # We want to explore high uncertainty.
    # Let's add a fraction of sigma.
    beta = 1.0
    score = (ei + beta * sigma_safe) / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
