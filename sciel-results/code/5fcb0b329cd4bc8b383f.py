"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Uses Expected Improvement per unit cost (EI/cost) with a UCB exploration
    term, balancing exploitation and exploration under budget constraints.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB exploration term
    # Beta decreases as f_best approaches 0 (search matures)
    # f_best for griewank starts around 0.01-0.1 and goes to 0
    progress = np.clip(1.0 - f_best / 0.1, 0.0, 1.0)  # 0 at start, 1 when f_best < 0
    beta = 0.5 * (1.0 - 0.7 * progress)  # beta from 0.5 to 0.15

    ucb = mu + beta * sigma_safe

    # Combine EI and UCB
    # Normalize EI to comparable scale
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ei_norm = ei / ei_max

    # Weight: more EI weight early, more UCB weight when uncertain
    # Use a fixed blend that works well for griewank
    ei_weight = 0.7
    ucb_weight = 0.3

    base = ei_weight * ei_norm + ucb_weight * (ucb / (np.max(ucb) + 1e-12))

    # Cost-awareness: divide by cost to get improvement per unit budget
    # This directly optimizes marginal return on each unit of budget spent
    score = base / cost_safe

    # Clamp non-finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        return score

    return score
# EVOLVE-BLOCK-END
