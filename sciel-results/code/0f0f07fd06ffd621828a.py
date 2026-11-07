import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate a dynamic reference point for Hypervolume Improvement
    # This encourages exploration in high-uncertainty regions by
    # setting the reference point slightly worse than the current best.
    max_sigma = np.max(sigma)
    # Avoid division by zero or extremely small numbers
    if max_sigma < 1e-6:
        ref = f_best
    else:
        # Use a factor of 1.0 to balance exploration and exploitation
        # This makes the acquisition function equivalent to EI with a
        # reference point that expands as uncertainty increases.
        ref = f_best + 1.0 * max_sigma
        
    z = (ref - mu) / (sigma + 1e-9)
    # Expected Improvement relative to ref
    scores = (ref - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
