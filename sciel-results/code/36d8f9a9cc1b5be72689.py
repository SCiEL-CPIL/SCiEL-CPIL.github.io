import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize the score: -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Ensure sigma is positive to avoid division by zero in other contexts, 
    # though here we just multiply.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Beta controls the exploration-exploitation trade-off.
    # A value around 1.0 to 2.0 is typical for 1D problems.
    beta = 1.5
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
