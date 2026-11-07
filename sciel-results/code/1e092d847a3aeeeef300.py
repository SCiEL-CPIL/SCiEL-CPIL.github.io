"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Cost-Weighted Probability of Improvement (CW-PoI).
    - PoI is more robust than EI to outliers and provides a smooth exploration-exploitation balance.
    - Dividing by cost strictly enforces the budget constraint, favoring cheaper evaluations.
    - This structural change moves away from the stage-adaptive EI/LCB hybrid to a single, 
      probabilistic criterion that naturally balances uncertainty and cost.
    """
    start_time = time.time()
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical overflow/underflow in CDF
    # For z > 10, Phi(z) is 1. For z < -10, Phi(z) is 0.
    z_clipped = np.clip(z, -10, 10)
    
    poi = norm.cdf(z_clipped)
    
    # Cost-Weighted Score
    scores = poi / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Fallback: If we are running out of time, return a simple heuristic
        # that favors low mu and low cost
        simple_scores = (f_best - mu) / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
