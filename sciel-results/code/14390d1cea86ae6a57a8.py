import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) acquisition function for minimization
    # Score = -mu + beta * sigma
    # Higher score indicates a more promising candidate.
    
    # Beta controls exploration (uncertainty) vs exploitation (mean)
    # A value of 3.0 is often effective for small sample sizes in 1D problems.
    beta = 3.0
    
    # Ensure sigma is safe (avoid division by zero or negative values if any)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate LCB scores
    # We want to minimize the function, so we look for low mu.
    # We want to explore high sigma.
    # LCB = mu - beta * sigma
    # We return -LCB so that higher values are better.
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
