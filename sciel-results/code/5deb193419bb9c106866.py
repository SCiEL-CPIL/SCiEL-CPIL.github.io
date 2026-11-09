"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 5s safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) for minimization.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        # Simple greedy fallback: minimize mu per unit cost
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB for minimization: we want to minimize mu - kappa * sigma
    # So score contribution is kappa * sigma - mu
    # We blend EI and UCB to balance exploitation and exploration
    # EI already captures improvement potential; UCB adds exploration pressure
    
    # Adaptive kappa: use a moderate exploration constant
    # In low-budget BO (total cost < 30), exploration is important
    kappa = 2.0
    
    # Combine: primary signal is cost-aware EI, secondary is cost-aware UCB
    # This gives us exploitation (EI) with exploration (UCB) bias
    ei_score = ei / cost_safe
    ucb_score = (kappa * sigma - mu) / cost_safe
    
    # Blend: weight EI more heavily since it directly measures improvement
    # but add UCB to encourage exploration of uncertain regions
    score = 0.7 * ei_score + 0.3 * ucb_score

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
