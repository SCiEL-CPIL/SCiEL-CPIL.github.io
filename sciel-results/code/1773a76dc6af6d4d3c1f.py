"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for time management
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we are too close to the limit
    elapsed = time.time() - _start_time
    if elapsed > 295:
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is strictly positive
    cost = np.maximum(cost, 1e-9)
    
    # z-score for Expected Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Structural change: Cost-normalized EI with relative cost efficiency.
    # Instead of dividing by an absolute cost penalty, compute a relative
    # cost efficiency score that rewards points cheaper than the median.
    # This makes the acquisition adaptive to the cost distribution of candidates.
    
    # Relative cost: normalize by the median cost in this batch
    med_cost = np.median(cost)
    # Cost efficiency: 1.0 at median, >1 for cheaper, <1 for more expensive
    cost_efficiency = med_cost / cost
    cost_efficiency = np.clip(cost_efficiency, 0.3, 3.0)
    
    # Exploration term: sigma-based uncertainty bonus
    exploration = sigma * 0.5
    
    # Combine: EI scaled by cost efficiency + exploration scaled by sqrt(cost_efficiency)
    # The product structure means cheap points get a multiplicative boost on both terms
    scores = ei * cost_efficiency + exploration * np.sqrt(cost_efficiency)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
