import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division errors
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma
    
    # Compute Probability of Improvement
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
