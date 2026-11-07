"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are 1-D arrays of length 2000
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Clip sigma to avoid division by zero and extreme values
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Expected Improvement per unit cost (cost-aware EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # cost is in range [1, 2] since mean(x scaled to [0,1]) is in [0,1]
    # But let's be safe and use cost as given
    cost_aware_ei = ei / cost
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps with coverage, especially in early iterations when sigma is high
    # The bonus is scaled to be bounded and diminishes as sigma decreases
    exploration_bonus = 0.1 * sigma / (sigma + 0.1)
    
    # Combine: cost-aware EI + bounded exploration bonus
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
