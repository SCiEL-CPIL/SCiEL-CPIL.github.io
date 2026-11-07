"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # seconds
_MARGIN = 5.0      # safety margin before deadline


def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: return cost-aware PI to be safe
        z = (f_best - mu) / (sigma + 1e-9)
        return norm.cdf(z) / (cost + 1e-9)

    # Clamp sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)

    # Standardize improvement
    z = (f_best - mu) / sigma

    # Probability of Improvement (PI): robust, less sensitive to variance inflation
    pi = norm.cdf(z)

    # Expected Improvement (EI): more aggressive exploitation
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Variance penalty: downweight regions with high uncertainty (noise chasing)
    # Use a soft penalty: 1 / (1 + alpha * sigma^2)
    alpha = 2.0
    var_penalty = 1.0 / (1.0 + alpha * sigma * sigma)

    # Combine PI and EI with variance penalty
    # Weight: 0.5 * PI + 0.5 * EI, both multiplied by variance penalty
    score = (0.5 * pi + 0.5 * ei) * var_penalty

    # Cost normalization: divide by cost to favor cheaper evaluations
    # cost is in [1, 2], so dividing by cost gives a range of [0.5, 1.0] multiplier
    score = score / (cost + 1e-9)

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
