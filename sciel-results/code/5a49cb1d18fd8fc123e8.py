import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Pure GP-LCB (Lower Confidence Bound) acquisition for minimization.
    score = -mu - kappa * sigma
    Higher score means more promising (lower predicted value with uncertainty).
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Fixed exploration parameter
    kappa = 2.0
    
    # LCB score: we want to minimize, so lower (-mu - kappa*sigma) is better
    # We return higher scores for better candidates, so:
    scores = -mu - kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
