"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Adaptive UCB for minimization
    # Higher score means more likely to be a better (lower) value
    # Score = -mu + beta * sigma
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive beta: decrease exploration as we converge
    # Use the spread of the posterior relative to current best
    # When sigma is small relative to (f_best - mu), we're exploiting
    # When sigma is large, we need more exploration
    
    # Compute a measure of how "tight" our knowledge is
    # If f_best - mu is large and positive, we're far from optimum -> explore more
    # If f_best - mu is small, we're close -> exploit more
    
    # Base beta that adapts to the uncertainty level
    # Use the median sigma as a reference scale
    med_sigma = np.median(sigma)
    
    # If median sigma is large relative to typical improvements, explore more
    # Typical improvement scale: how much mu varies
    mu_range = np.max(mu) - np.min(mu) if np.max(mu) > np.min(mu) else 1e-6
    
    # Adaptive beta: start around 1.5-2.0, decrease as we converge
    # Use a soft adaptation based on the posterior spread
    # When med_sigma is small (we're confident), reduce beta
    # When med_sigma is large (uncertain), keep beta higher
    
    # Normalize sigma by the scale of the problem
    # f_best is around -1.8, mu values are in similar range
    # Use a characteristic scale
    scale = max(abs(f_best), 1e-6)
    
    # Relative uncertainty
    rel_uncert = med_sigma / scale
    
    # Beta adapts: high uncertainty -> high beta, low uncertainty -> lower beta
    # Clamp to reasonable range
    beta = np.clip(2.0 * rel_uncert + 0.3, 0.3, 3.0)
    
    # Additionally, for points where sigma is very small but mu is close to f_best,
    # we want to exploit. For points with high sigma, we want to explore.
    # The UCB naturally does this, but we can enhance it.
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
