"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: balance expected improvement against cost.
    # Use a small epsilon to avoid division by zero.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost, but add a small constant to avoid extreme values
    # Also, we can use a power of cost to make it more sensitive to high costs
    cost_safe = np.maximum(cost, 1.0)
    
    # Return EI / cost
    return ei / cost_safe
# EVOLVE-BLOCK-END
