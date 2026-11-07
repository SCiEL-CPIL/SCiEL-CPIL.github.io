"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) with a small exploration boost.
    
    PI(x) = P(f(x) < f_best | mu, sigma) = Phi((f_best - mu) / sigma)
    
    We add a small multiple of sigma to prevent premature convergence in
    multimodal landscapes where PI can become zero in high-uncertainty regions.
    The boost is scaled relative to the spread of mu to be adaptive.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standard Probability of Improvement
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Small exploration boost proportional to local uncertainty
    # Scale the boost by the median sigma to make it adaptive
    sigma_median = np.median(sigma_safe)
    boost_scale = 0.1 * sigma_median
    
    scores = pi + boost_scale * sigma_safe / (sigma_median + 1e-12)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
