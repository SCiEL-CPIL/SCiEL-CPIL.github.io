"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: uses Probability of Improvement (PI) with a soft
    cost penalty via exponentiation, plus a small sigma-scaled exploration bonus.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    t_start = time.time()
    
    # Convert inputs to arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: Phi(z)
    pi = norm.cdf(z)
    
    # Ensure PI is in [0, 1]
    pi = np.clip(pi, 1e-10, 1.0)
    
    # Soft cost penalty via exponentiation: PI^(1/cost)
    # This naturally prioritizes cheap, high-confidence improvements
    # When cost=1, exponent=1, so PI stays the same
    # When cost=2, exponent=0.5, so PI is reduced (square root)
    cost_exponent = 1.0 / cost
    pi_cost_aware = np.power(pi, cost_exponent)
    
    # Add a small exploration bonus for high uncertainty
    # This encourages exploring regions with high sigma, but scaled by cost
    exploration_bonus = 0.05 * sigma / cost
    
    # Combine: cost-aware PI + exploration bonus
    scores = pi_cost_aware + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > 320:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
