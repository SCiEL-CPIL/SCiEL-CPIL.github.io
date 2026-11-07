"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) based acquisition function.
    
    POI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    We scale by the magnitude of expected gain to balance exploration/exploitation.
    Using (f_best - mu) * POI gives a score that is high when:
    - The mean is well below f_best (exploitation)
    - There is high probability of improvement (exploration via uncertainty)
    
    This differs from EI by removing the sigma*pdf(z) term, making it more
    focused on the probability mass rather than expected magnitude, but the
    scaling by (f_best - mu) ensures we still prefer larger gains.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Scale by expected improvement magnitude (f_best - mu)
    # This ensures we prefer points that are likely to be much better
    score = (f_best - mu) * poi
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
