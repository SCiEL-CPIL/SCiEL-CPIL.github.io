"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    with a cost-weighted uncertainty term.
    
    POI = Phi((f_best - mu) / sigma)
    
    We use a log-barrier formulation:
    Score = log(1 + POI) * (sigma^gamma) / (cost^delta)
    
    - log(1 + POI) smoothly saturates exploitation, preventing over-focusing
      on near-certain improvements and encouraging exploration.
    - sigma^gamma with gamma > 1 emphasizes high-uncertainty regions.
    - Dividing by cost^delta penalizes expensive evaluations.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > 310.0:
        # Fallback: simple sigma-based score normalized by cost
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        fallback = np.maximum(sigma_arr, 1e-8) / np.maximum(cost_arr, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for POI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    POI = norm.cdf(z)
    
    # Log-barrier on POI to smooth out extreme values
    log_poi = np.log1p(POI)
    
    # Uncertainty term: sigma^gamma to emphasize exploration
    gamma = 1.5
    uncertainty = np.power(sigma_safe, gamma)
    
    # Cost penalty: cost^delta
    delta = 1.0
    cost_penalty = np.power(cost_safe, delta)
    
    # Combine
    scores = log_poi * uncertainty / cost_penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
