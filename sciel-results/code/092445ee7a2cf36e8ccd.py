"""Cost-aware acquisition for Bayesian optimization on shekel 4d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and bail out with a safe fallback if needed.
_DEADLINE = 320.0  # safety margin
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Uses a cost-aware variant of Expected Improvement with an exploration
    bonus that balances exploitation (low mu) with exploration (high sigma)
    while penalizing expensive evaluations.
    """
    if _check_deadline():
        # Emergency fallback: return something reasonable (exploit best)
        return -mu / (cost + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost normalization: cost is in range [1, 2]
    # Use log-cost penalty to be less aggressive
    cost_penalty = np.log(cost + 1e-9)

    # Cost-aware EI: divide by cost (linear penalty)
    # This encourages finding good values cheaply
    score = ei / cost

    # Add a small exploration bonus for high uncertainty
    # This helps escape local optima
    exploration_bonus = 0.05 * sigma / cost

    score += exploration_bonus

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
