import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    """
    # Ensure sigma is non-negative and has a floor
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # LCB for minimization: mu - kappa * sigma
    # We want to minimize this, so we maximize kappa * sigma - mu
    kappa = 2.0  # Exploration-exploitation trade-off
    
    # Calculate LCB-based score
    lcb_score = kappa * sigma_safe - mu
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-6)
    scores = lcb_score / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
