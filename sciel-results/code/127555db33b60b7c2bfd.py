"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t_start = time.time()
    
    # Expected improvement per unit cost
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    scores = ei / cost
    
    # Add a small exploration term to avoid premature convergence
    # Use probability of improvement (POI) as a secondary signal
    poi = norm.cdf(z)
    
    # Combine EI and POI with a balance parameter
    balance = 0.5
    scores = balance * ei / cost + (1 - balance) * poi / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
