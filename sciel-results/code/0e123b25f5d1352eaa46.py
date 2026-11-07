# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    Uses Expected Improvement (EI) per unit cost.
    Includes a deadline guard to switch to a simple heuristic if time is running out.
    """
    # Deadline guard: if we are close to the time limit, return simple scores
    # This prevents timeouts and ensures a valid return
    if time.time() - _START_TIME > _TIME_LIMIT:
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return 1.0 / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to avoid division by zero
    sigma = np.clip(sigma, 1e-9, None)
    cost = np.clip(cost, 1e-9, None)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
