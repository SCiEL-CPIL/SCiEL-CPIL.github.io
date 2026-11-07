import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    This is a structural change from the normalized EI/UCB blend to a raw,
    unnormalized LCB which preserves the scale of the objective function.
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # LCB score: -mu + beta * sigma
    # We maximize this score.
    # beta controls exploration.
    # A value of 2.0-3.0 is often effective for low-budget BO.
    beta = 2.0
    
    scores = -mu + beta * sigma_safe
    
    # Handle NaNs and Infs
    # If a point has infinite uncertainty, it should be prioritized?
    # np.nan_to_num with posinf=np.inf keeps inf as inf.
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
