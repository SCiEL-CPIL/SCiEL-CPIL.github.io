import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    """
    # Exploration parameter
    beta = 1.5
    
    # Ensure sigma is not zero to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the potential for improvement per unit cost.
    # Potential = beta * sigma - mu
    potential = beta * sigma_safe - mu
    
    # Cost-aware score
    scores = potential / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
