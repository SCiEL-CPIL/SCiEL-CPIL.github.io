import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize the score: beta * sigma - mu
    # This encourages exploration of uncertain regions (high sigma)
    # and exploitation of promising regions (low mu).
    
    # Ensure sigma is non-negative to avoid numerical issues
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Exploration-exploitation trade-off parameter
    # A value of 1.0 is a standard heuristic for UCB
    beta = 1.0
    
    # Calculate UCB scores
    scores = beta * sigma_safe - mu
    
    # Ensure finite scores (mu and sigma should be finite from GP)
    return scores
