"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Standard Expected Improvement for minimization
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Clip to non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Use Square Root-EI (Power-EI with p=0.5)
    # This provides a smoother compression of the EI values compared to Log-EI,
    # potentially maintaining better discrimination between high-EI candidates
    # while still suppressing the influence of zero-improvement regions.
    scores = np.sqrt(ei + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
