import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, where high score indicates a promising point.
    # For minimization, we prefer low mu and high sigma.
    # Score = -mu + beta * sigma
    
    # Choose a beta parameter. 
    # A fixed beta is used as the number of iterations is not passed.
    # beta = 1.0 is a reasonable default for balancing exploration and exploitation.
    beta = 1.0
    
    # Calculate UCB score
    # Note: sigma is standard deviation.
    # We ensure sigma is non-negative, though it should be by definition.
    sigma_safe = np.maximum(sigma, 1e-10)
    
    score = -mu + beta * sigma_safe
    
    return score
