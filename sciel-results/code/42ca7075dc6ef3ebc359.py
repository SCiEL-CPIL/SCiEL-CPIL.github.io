"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
# Set a deadline slightly before the hard kill to allow for safe return
_DEADLINE = _start_time + 322.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        scores = -mu / np.where(cost > 1e-9, cost, 1.0)
        return np.where(np.isfinite(scores), scores, 0.0)

    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    sigma_safe = sigma + 1e-9

    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)

    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)

    # A slightly different cost-aware formulation: use a geometric blend
    # between EI and EI/cost to soften the cost penalty. This preserves the
    # design family while adjusting the exploitation/exploration balance
    # relative to pure EI/cost.
    scores = 0.5 * ei + 0.5 * (ei / cost_safe)

    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
