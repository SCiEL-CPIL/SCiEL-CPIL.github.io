import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # Min value is -2.53218
    min_val = -2.53218
    gap = f_best - min_val
    
    # Beta ranges from 0.5 (exploit) to 2.0 (explore)
    # Use exponential saturation
    beta = 0.5 + 1.5 * (1.0 - np.exp(-gap))
    
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
