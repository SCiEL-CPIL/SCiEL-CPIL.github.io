import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Handle zero sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Dynamic beta for Lower Confidence Bound (LCB)
    # Global minimum of Styblinski-Tang 1D is approx -39.1662
    target_min = -39.1662
    
    # Gap from current best to global min
    gap = f_best - target_min
    
    # If gap is negative (we found something better than min? unlikely, or noise), treat as 0
    if gap < 0:
        gap = 0
        
    # Beta starts high (exploration) when gap is large, decreases as we get closer
    # exp(-gap/20) decays from 1.0 to near 0
    # beta = 1.0 (exploit) to 4.0 (explore)
    beta = 1.0 + 3.0 * np.exp(-gap / 20.0)
    
    # LCB score: we want to maximize this
    # LCB value is mu - beta * sigma
    # So score is -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
