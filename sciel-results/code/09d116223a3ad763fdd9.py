import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Avoid division by zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PoI)
    # P(f(x) < f_best) = P(N(mu, sigma) < f_best) = CDF((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Cost-aware score: PoI per unit cost
    # Higher score is better
    scores = poi / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
