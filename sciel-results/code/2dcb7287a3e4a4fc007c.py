import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration weight
    # High when f_best is large (early, poor estimate), low when f_best is small (late, good estimate)
    # Using a saturating function to keep beta in [0, 1)
    beta = f_best / (1.0 + f_best)
    
    # UCB-like term: beta * sigma
    # We add this to EI to encourage exploration in high uncertainty regions
    # especially when the current best is not very good (f_best is large)
    scores = ei + beta * sigma
    
    return scores
