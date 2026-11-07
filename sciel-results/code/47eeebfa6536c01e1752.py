"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Probability of Improvement
    # z = (f_best - mu) / sigma
    # POI = Phi(z)
    # Score = POI / cost
    
    # Add a small epsilon to sigma to avoid division by zero
    sigma_safe = sigma + 1e-9
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Cost-aware score
    scores = poi / cost
    
    return scores
# EVOLVE-BLOCK-END
