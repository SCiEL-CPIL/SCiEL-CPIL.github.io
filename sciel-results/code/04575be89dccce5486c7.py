"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement with milder cost normalization
    # Cost is in [1, 2]. We use a penalty that varies from 1 to 1.5.
    # This reduces the bias towards cheap points, allowing exploration of expensive regions.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Milder cost penalty: 1 + 0.5 * (cost - 1)
    # If cost=1, penalty=1. If cost=2, penalty=1.5.
    penalty = 1.0 + 0.5 * (cost - 1.0)
    
    scores = ei / (penalty + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
