import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    GP-Lower Confidence Bound (GP-LCB) acquisition function for minimization.
    Returns a score where higher values indicate more promising points.
    score = -mu + beta * sigma
    """
    # Exploration coefficient
    beta = 2.0
    
    # Calculate LCB score (higher is better)
    # We want to minimize mu - beta * sigma, so we maximize -(mu - beta * sigma)
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
