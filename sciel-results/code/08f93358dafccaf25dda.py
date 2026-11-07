"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
# 330s wall clock limit, keep 5s margin for safety
_DEADLINE = _START_TIME + 325

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PoI).
    
    This approach prioritizes points with a high probability of improving upon
    the current best value, normalized by the evaluation cost. This structural
    choice has shown superior performance on multimodal landscapes like Rastrigin
    compared to Expected Improvement (EI) by balancing exploration and exploitation
    differently.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline to ensure we return within time limits
    if time.time() > _DEADLINE - 5:
        # Fallback: prefer cheap points if we are running out of time
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe

    # Sanitize inputs to prevent numerical issues
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to a reasonable range to ensure numerical stability of the CDF
    # norm.cdf(-10) is effectively 0, norm.cdf(10) is effectively 1
    # Clipping prevents extreme values from causing floating point issues
    z = np.clip(z, -10.0, 10.0)
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Cost-aware: PoI per unit cost
    # We want to maximize this score
    scores = poi / cost_safe
    
    # Ensure all scores are finite to satisfy the task contract
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
