"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PI)
    and Expected Improvement (EI), normalized by cost.
    
    mu: shape (2000,) - GP mean
    sigma: shape (2000,) - GP std
    f_best: float - best observed value
    cost: shape (2000,) - evaluation costs
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Combine EI and PI with a blend
    # Weight toward PI for robustness in high dimensions, but keep EI for exploitation
    blend = 0.5 * ei + 0.5 * pi * sigma_safe
    
    # Normalize by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = blend / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
