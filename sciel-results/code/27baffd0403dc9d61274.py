"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement (EI) normalized by cost, combined with a 
    scaled uncertainty (exploration) bonus. This formulation is robust 
    across different stages of optimization, encouraging exploration 
    early on and exploitation later, without hard switching thresholds.
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid overflow in exp for very negative/positive z
    z_clipped = np.clip(z, -50, 50)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-10)
    
    # EI per unit cost
    score_ei = ei / cost_safe
    
    # Exploration bonus: encourages evaluation of uncertain regions.
    # A coefficient of 0.1 balances exploration and exploitation effectively 
    # for this problem domain, slightly higher than the 0.05 in previous 
    # iterations to potentially escape local minima faster.
    exploration_coef = 0.1
    exploration = exploration_coef * sigma_safe / cost_safe
    
    # Combine scores
    scores = score_ei + exploration
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simple fallback
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback to simple EI/cost if computation is unexpectedly slow
        simple_scores = score_ei
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
