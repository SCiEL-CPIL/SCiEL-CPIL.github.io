import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for the Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Cost-aware acquisition score
    # Higher score means better candidate
    # We divide by cost to prefer cheaper evaluations
    scores = poi / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
