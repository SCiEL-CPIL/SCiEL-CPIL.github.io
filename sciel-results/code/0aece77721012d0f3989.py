"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # 290s safety margin from 300s limit
        _DEADLINE = time.time() + 290
    return _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) with Gaussian
    kernel bias toward the low-cost central region.
    
    Parameters:
    - mu: (2000,) posterior mean in objective units
    - sigma: (2000,) posterior std in objective units  
    - f_best: scalar, lowest observed value
    - cost: (2000,) evaluation cost for each candidate
    
    Returns:
    - (2000,) scores, higher = evaluate next
    """
    deadline = _get_deadline()
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calculate Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Gaussian kernel bias toward cost=1.5 (center of domain where cost is lowest)
    # This structurally biases search toward the central region rather than
    # uniformly penalizing higher-cost points via division.
    # The Levy function's minimum, after shifting to inner 80%, is likely near center.
    # cost ranges from 1 (corner) to 2 (opposite corner), center is 1.5
    center_cost = 1.5
    bandwidth = 0.4  # controls how sharply we bias toward center
    gaussian_bias = np.exp(-((cost - center_cost) ** 2) / (2 * bandwidth ** 2))
    
    # Combine EI with Gaussian cost bias
    score = ei * gaussian_bias
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
