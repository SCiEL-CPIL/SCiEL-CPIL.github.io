import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Gaussian Process-Upper Confidence Bound (GP-UCB) for minimization
    # We maximize the score: -mu + beta * sigma
    # The directive suggests a time-varying beta or a fixed high value like 3.0.
    # Given the low budget (30 evaluations) and the inspiration that beta=2.5 worked well,
    # we use beta=3.0 to encourage robust exploration.
    
    # Handle zero or near-zero sigma to avoid numerical instability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Fixed exploration parameter beta
    # The directive suggests approximating t from f_best or using a fixed value.
    # Using a fixed value of 3.0 as per the directive's suggestion for a "high value".
    beta = 3.0
    
    # UCB score for minimization: we want low mu and high sigma
    # score = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
