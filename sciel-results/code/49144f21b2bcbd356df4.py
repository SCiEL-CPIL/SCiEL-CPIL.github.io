import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - kappa * sigma
    # Since we must return a score where HIGHER is better,
    # we maximize the negative of the LCB: -mu + kappa * sigma
    
    kappa = 2.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    score = -mu + kappa * sigma
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
