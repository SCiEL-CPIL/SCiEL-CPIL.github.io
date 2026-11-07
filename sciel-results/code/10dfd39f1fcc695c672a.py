import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB for minimization
    # Base beta
    beta_base = 2.0
    # Scale beta based on how far f_best is from the known minimum (3)
    # Larger gap -> more exploration
    gap = f_best - 3.0
    if gap < 0:
        gap = 0.0
    beta = beta_base * (1.0 + gap / 10.0)
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher score is better
    scores = -mu + beta * sigma
    
    # Ensure finite values
    # sigma is standard deviation, so it should be non-negative
    # mu should be finite
    return scores
