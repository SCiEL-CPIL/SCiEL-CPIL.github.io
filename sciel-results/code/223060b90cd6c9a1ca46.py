"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with a small exploitation parameter xi.
    # xi > 0 makes the acquisition function more greedy, preferring points
    # that are very likely to be better than the current best.
    xi = 0.1
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement
    improvement = f_best - mu - xi
    
    # Calculate z-score
    z = improvement / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to 0 (standard practice, though mathematically EI >= 0)
    ei = np.maximum(ei, 0)
    
    # Divide by cost to make it cost-aware
    # cost is 1 + mean(x scaled to [0,1]), so it's in [1, 2]
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
