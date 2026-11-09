"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of Expected Improvement (EI)
    and Expected Reduction in Uncertainty (ERU), normalized by cost.
    
    We use a hybrid approach:
    1. Compute Expected Improvement (EI) which drives exploitation.
    2. Add a small exploration term based on entropy of the posterior (or simply sigma)
       to encourage exploring regions with high uncertainty, but dampened to avoid
       pure random search.
    3. Divide by cost to account for the cost-aware requirement.
    
    To ensure finite values and numerical stability, we clip sigma and use safe operations.
    """
    # Ensure sigma is positive and not too small to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z to avoid extreme values in CDF/PDF
    z_clipped = np.clip(z, -10, 10)
    
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Exploration term: we use a weighted sigma to encourage exploring uncertain regions
    # but not too much. A common heuristic is to add a fraction of sigma.
    # Let's use a small weight like 0.1 * sigma to balance exploration and exploitation.
    # Alternatively, we could use the differential entropy of the Gaussian, but
    # for simplicity and speed, we'll use a linear term.
    exploration = 0.1 * sigma_safe
    
    # Combine EI and exploration
    score = ei + exploration
    
    # Cost-aware: divide by cost
    # Ensure cost is not zero or too small
    cost_safe = np.maximum(cost, 1e-8)
    
    result = score / cost_safe
    
    # Ensure all values are finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
