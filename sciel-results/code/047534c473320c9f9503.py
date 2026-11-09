import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x), so we look for low mu and high sigma (uncertainty)
    # Score = -mu + beta * sigma
    # Higher score means "more promising" (lower expected value with penalty for uncertainty)
    
    # Clip sigma to avoid numerical issues if sigma is extremely small or large
    # Though sigma should be > 0.
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls exploration vs exploitation
    # 2.0 is a common choice for aggressive exploration in early BO
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
