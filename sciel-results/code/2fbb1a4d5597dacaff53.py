"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time to avoid exceeding 300s per evaluation
_start_time = time.time()
_DEADLINE_MARGIN = 280.0  # Leave 20s margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: maximize P(f(x) < f_best - eps) / cost.
    
    Parameters:
    - mu: (2000,) mean predictions in objective units
    - sigma: (2000,) std predictions in objective units
    - f_best: scalar, best observed value
    - cost: (2000,) evaluation costs
    
    Returns:
    - (2000,) acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_MARGIN:
        # Fallback: return uniform-ish scores to avoid hanging
        return np.ones_like(mu) / np.maximum(cost, 1e-6)
    
    # Small epsilon to focus on meaningful improvements
    eps = 0.01
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute z-score for threshold f_best - eps
    z = (f_best - eps - mu) / sigma_safe
    
    # Probability that f(x) < f_best - eps
    prob_improve = norm.cdf(z)
    
    # Clip to avoid numerical issues
    prob_improve = np.clip(prob_improve, 1e-10, 1.0 - 1e-10)
    
    # Cost-aware score: probability per unit cost
    score = prob_improve / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
