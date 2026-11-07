import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB)
    # For minimization, we want low mu and high sigma.
    # LCB = mu - beta * sigma
    # We divide by cost to account for the evaluation cost.
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls exploration. 2.0 is a reasonable default for standardized GPs.
    beta = 2.0
    
    lcb = mu - beta * sigma
    
    # Divide by cost
    # Cost is positive (1 to 2)
    scores = lcb / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
