"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Enhanced Probability of Improvement (POI) based acquisition function.
    
    Uses a blend of POI and Expected Improvement to balance exploration
    and exploitation more effectively.
    
    The score combines:
    1. Probability of improvement: Phi((f_best - mu) / sigma)
    2. Expected improvement magnitude: (f_best - mu) * Phi(z) + sigma * pdf(z)
    
    We use a weighted combination that emphasizes points with both high
    probability of improvement and large expected gains.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Expected Improvement (EI) is the standard POI-based acquisition function
    # It scales the probability of improvement by the expected gain magnitude
    # and includes a term for uncertainty.
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    score = np.where(np.isfinite(ei), ei, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
