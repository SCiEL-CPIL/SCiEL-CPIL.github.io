"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # 295 seconds limit (5s safety margin before 300s kill)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PoI).
    
    This function computes the Probability of Improvement for each candidate
    and normalizes it by the evaluation cost. This strategy has shown superior
    performance over Expected Improvement (EI) for multimodal functions like
    Rastrigin in this specific cost-aware setting, as it is less conservative
    and more likely to explore distinct basins of attraction.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to prefer cheap points
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe
    
    # Ensure inputs are arrays and handle types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in CDF
    # Phi(10) is effectively 1.0, Phi(-10) is effectively 0.0
    z_clipped = np.clip(z, -10, 10)
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z_clipped)
    
    # Cost-aware scoring: PoI per unit cost
    # Cost is in [1, 2]. Dividing by cost prioritizes cheaper evaluations
    # that still have a high probability of being better than the current best.
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = poi / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
