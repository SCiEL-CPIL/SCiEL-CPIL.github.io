"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses cost-normalized Expected Improvement with adaptive exploration.
    """
    t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - in objective units
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative (EI should be >= 0, but numerical issues can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive beta: higher when uncertainty is large relative to improvement
    # This encourages more exploration early on
    improvement = f_best - mu
    # Scale: when improvement is small relative to sigma, increase exploration
    beta = 0.5 * np.exp(-np.maximum(improvement, 0) / (sigma_safe + 1e-9))
    beta = np.clip(beta, 0.1, 1.0)
    
    # UCB-like exploration term
    ucb = -mu + beta * sigma_safe
    
    # Combine: primarily EI with some UCB, both cost-normalized
    # Weight: 0.8 for EI, 0.2 for UCB (tune the exploration weight)
    score = (0.8 * ei + 0.2 * ucb) / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    if time.time() - t0 > 280:
        return ei / cost_safe
    
    return score
# EVOLVE-BLOCK-END
