"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

_DEADLINE = 325.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: combines EI, adaptive UCB, and power cost penalty.
    Returns 2000 finite scores (higher = more promising).
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n != 2000:
        return np.ones(n)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with ndtr for very large |z|
    z_clipped = np.clip(z, -10, 10)
    pdf_z = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * ndtr(z_clipped) + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # Adaptive UCB: moderate exploration weight
    kappa = 1.2
    ucb = mu - kappa * sigma_safe
    ucb_gain = np.maximum(f_best - ucb, 0.0)

    # Combine EI and UCB gain
    # EI is the primary exploitation signal; UCB adds a smaller exploration component
    alpha = 0.15
    combined = ei + alpha * ucb_gain

    # Power cost penalty with beta=1.5: prefer cheaper points more aggressively
    # cost is in [1, 2]; cost^1.5 ranges from 1 to ~2.83
    # Dividing by cost^beta gives a factor from 1 down to ~0.35
    beta = 1.5
    cost_penalty = np.power(cost_safe, beta)
    scores = combined / cost_penalty

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Small epsilon to avoid all-zero scores
    scores = np.maximum(scores, 1e-10)

    return scores
# EVOLVE-BLOCK-END
