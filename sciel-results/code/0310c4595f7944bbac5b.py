"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we've been running too long, return a simple, fast score
    # to avoid being killed. We track time via a module-level variable.
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    elapsed = time.time() - acquisition._start_time
    if elapsed > 280:
        # Fallback: simple EI per cost, very fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Cost-aware acquisition: combine expected improvement with a penalty
    # for known-good regions to encourage exploration.
    #
    # Strategy:
    # 1. Compute standard Expected Improvement (EI)
    # 2. Compute a "novelty" or "uncertainty" bonus that rewards areas
    #    where sigma is high relative to the range of mu
    # 3. Penalize areas where mu is close to f_best AND sigma is low
    #    (already well-exploited)
    # 4. Divide by cost to make it cost-aware

    # Standard EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)

    # Uncertainty bonus: reward high sigma regions
    # Scale sigma to [0, 1] range for the bonus
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-12 else 1.0
    sigma_norm = sigma / sigma_max

    # Exploration bonus: high where sigma is high
    exploration_bonus = sigma_norm * 0.5

    # Penalty for well-exploited good regions:
    # When mu is close to f_best (good) and sigma is low (well-known),
    # we don't want to evaluate there again.
    # mu_range = f_best - mu (how much improvement we'd expect)
    mu_range = np.maximum(f_best - mu, 0.0)
    mu_range_max = np.max(mu_range) if np.max(mu_range) > 1e-12 else 1.0
    mu_range_norm = mu_range / mu_range_max

    # Penalty: high when mu_range is small (mu close to f_best) AND sigma is small
    # This is a "known-good" region penalty
    known_good_penalty = (1.0 - mu_range_norm) * (1.0 - sigma_norm) * 0.3

    # Combined score: EI + exploration bonus - known-good penalty
    # We want higher scores for candidates we should evaluate next
    score = ei + exploration_bonus - known_good_penalty

    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Cost-aware: divide by cost (higher cost = lower priority)
    cost_safe = np.maximum(cost, 1e-9)
    score = score / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
