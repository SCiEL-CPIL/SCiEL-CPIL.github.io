"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with a small safety margin.
    # Use np.clip to avoid division by zero and ensure finite outputs.
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Divide by cost, ensuring cost is at least a small positive value
    cost_safe = np.maximum(cost, 1e-8)
    
    return ei / cost_safe
# EVOLVE-BLOCK-END
