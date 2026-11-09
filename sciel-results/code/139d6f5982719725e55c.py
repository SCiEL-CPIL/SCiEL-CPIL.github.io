import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate z-scores
    # sigma is standard deviation
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost penalty
    # cost is in [1, 2]
    # We want to penalize higher costs.
    # Normalize cost penalty to [0, 1] range relative to min cost 1.
    # penalty = cost - 1.0
    # Scale factor lambda
    # If lambda is 1, a cost of 2 reduces score by 1.
    # PI is in [0, 1].
    # This means a point with PI=1 (certain improvement) and cost=2 gets score 0.
    # A point with PI=0.5 and cost=1 gets score 0.5.
    # This seems reasonable.
    
    lambda_cost = 1.0
    penalty = lambda_cost * (cost - 1.0)
    
    scores = pi - penalty
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
