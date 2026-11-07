"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with exploration bonus.
    # Use a small exploration term to avoid getting stuck in local optima.
    # The cost is in [1, 2] since mean(x) in [0,1] gives cost in [1,2].
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps when sigma is large (uncertain regions)
    exploration = 0.1 * sigma_safe
    
    # Cost-aware: divide by cost, add exploration
    score = ei / cost + exploration / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
