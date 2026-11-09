"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement with exploration parameter xi, normalized by cost.
    # xi > 0 encourages exploration of regions with potential for significant improvement.
    xi = 0.1
    
    # Protect against division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for the modified improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # Calculate Expected Improvement
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize by cost to prefer cheaper evaluations
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    return ei / cost_safe
# EVOLVE-BLOCK-END
