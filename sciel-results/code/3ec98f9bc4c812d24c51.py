"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Uses Expected Improvement per unit cost (EI / cost) as the core objective,
    with a diminishing UCB exploration term that adapts to search progress.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Core: Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB component for exploration
    # Beta decreases as we find better solutions (f_best approaches 0)
    progress = np.clip(f_best, 0, 1)
    beta = 0.3 * (1.0 + progress)
    ucb = -mu + beta * sigma_safe

    # Blend EI and UCB: primarily EI with a diminishing UCB contribution
    # Scale UCB relative to EI to keep them comparable
    ei_median = np.median(ei)
    if ei_median > 1e-9:
        ucb_scale = 0.15
    else:
        ucb_scale = 0.0

    base_score = ei + ucb_scale * ucb

    # Cost-aware: divide by cost to get EI per unit cost
    # This directly optimizes marginal return per unit of budget
    score = base_score / cost_safe

    # Clamp non-finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        return ei / cost_safe

    return score
# EVOLVE-BLOCK-END
