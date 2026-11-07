"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 310.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Knowledge Gradient (EI) normalized by cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = better candidate to evaluate)
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement (Knowledge Gradient for 1-step)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI is non-negative by definition for minimization
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness:
    # Maximize efficiency: Improvement per unit cost.
    # cost is in [1, 2].
    # Dividing by cost directly penalizes expensive evaluations.
    
    # Exploration term:
    # Add a small multiple of sigma to encourage visiting uncertain regions,
    # even if immediate EI is low, but also penalize this exploration by cost.
    # This helps escape local optima, especially if the global minimum is in a high-cost region.
    
    beta = 0.1
    
    # Combined score: (EI + Exploration) / Cost
    scores = (ei + beta * sigma_safe) / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
