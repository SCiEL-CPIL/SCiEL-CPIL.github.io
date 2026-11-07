import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: PI minus a penalty proportional to cost
    # Cost is typically in [1, 2]
    # PI is in [0, 1]
    # Lambda balances exploration (PI) vs cost avoidance
    lambda_cost = 0.5
    scores = pi - lambda_cost * cost
    
    # Replace non-finite values with -inf to ensure they are not selected
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
