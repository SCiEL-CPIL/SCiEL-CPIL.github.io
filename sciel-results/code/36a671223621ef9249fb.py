"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling with cost normalization
    # Sample a single theta from N(mu, sigma) for each candidate
    # Score = -theta / cost (higher is better, so we want low theta / low cost)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample theta for each candidate
    theta = mu + sigma_safe * np.random.standard_normal(mu.shape)
    
    # Cost normalization: prefer lower cost
    # Score = -theta / cost
    # Higher score means lower theta and/or lower cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = -theta / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
