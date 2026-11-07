import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB Acquisition
    # Tuned beta for Beale function optimization
    beta = 2.0
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    return scores
