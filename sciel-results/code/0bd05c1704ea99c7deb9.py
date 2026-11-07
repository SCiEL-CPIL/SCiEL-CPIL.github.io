"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses Expected Improvement (EI) normalized by cost, with a small UCB exploration term.
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    # cost is 1 + mean(x_scaled), so cost is in [1, 2]
    # We want to penalize high cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Primary score: EI per unit cost
    score = ei / cost_safe
    
    # Add a small exploration bonus (UCB-like) to avoid getting stuck
    # The GP is well-calibrated for Griewank, so we don't need much exploration
    # but a tiny bit helps escape local minima in the GP surface
    exploration = 0.01 * sigma_safe / cost_safe
    
    # Combine
    final_score = score + exploration
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
