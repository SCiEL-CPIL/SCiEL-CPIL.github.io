import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We maximize the score: -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma)
    
    beta = 1.5
    
    # Handle numerical stability for sigma
    # If sigma is 0, the exploration term is 0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
