"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 290  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check time budget - if we're running low, return a simple score
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback: just use -mu/cost
        return -mu / np.maximum(cost, 1e-9)
    
    # Cost-aware Expected Improvement with small exploitation parameter
    # Use xi=0.02 for a balance between exploration and exploitation
    xi = 0.02
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the improvement over current best (with exploitation bias)
    improvement = f_best - mu - xi
    
    # Calculate z-score, clip to avoid numerical issues with extreme values
    z = improvement / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to 0
    ei = np.maximum(ei, 0.0)
    
    # Divide by cost to make it cost-aware
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
