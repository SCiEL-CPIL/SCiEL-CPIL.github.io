import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Look-Ahead Expected Improvement (LAEI) approximation.
    Combines standard Expected Improvement with an exploration term
    proportional to the standard deviation, which approximates the
    information gain (lookahead value) of sampling uncertain points.
    """
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term (Look-Ahead proxy)
    # Higher sigma means more potential to update the global model
    # and reveal better regions.
    exploration_weight = 0.5
    exploration_term = exploration_weight * sigma_safe
    
    # Combined Score
    scores = ei + exploration_term
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
