"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Log-Probability of Improvement
    with a cost-weighted uncertainty term.
    
    Formula: Score = log(1 + PI) * (sigma^gamma) / (cost^delta)
    
    - log(1 + PI) smoothly saturates exploitation, preventing over-focusing 
      on near-certain improvements and encouraging exploration.
    - sigma^gamma with gamma > 1 emphasizes high-uncertainty regions more strongly.
    - cost^delta with delta < 1 softens the cost penalty, allowing the algorithm 
      to consider expensive points if the potential gain is high enough.
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
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    # PI = P(f(x) < f_best) = Phi(z)
    PI = norm.cdf(z)
    
    # Parameters for the structural change
    gamma = 1.5  # Emphasize uncertainty
    delta = 0.5  # Soften cost penalty
    
    # Log-barrier on PI to prevent over-exploitation
    # log(1 + PI) is in [0, log(2)]
    log_PI = np.log1p(PI)
    
    # Uncertainty term
    # sigma^gamma
    unc_term = sigma_safe ** gamma
    
    # Cost term
    # 1 / cost^delta
    cost_term = 1.0 / (cost_safe ** delta)
    
    scores = log_PI * unc_term * cost_term
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
