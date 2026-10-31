"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for estimating fill probability of remainders
# These are tuned for typical Weibull sequences with capacity 100
_WEIBULL_SHAPE = 1.5
_WEIBULL_SCALE = 20.0


def _weibull_cdf(x, shape=_WEIBULL_SHAPE, scale=_WEIBULL_SCALE):
    """Compute Weibull CDF: P(X <= x) = 1 - exp(-(x/scale)^shape)"""
    x = np.maximum(x, 0.0)
    return 1.0 - np.exp(-np.power(x / scale, shape))


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Weibull-regret-based best-fit.
    We combine best-fit (minimize remaining_after) with a regret term based on
    the Weibull CDF of the remaining space. The idea is that a remainder 'r'
    has value proportional to the probability that a future Weibull item can
    fit in it, which is F(r) = 1 - exp(-(r/scale)^shape).
    
    Score = -remaining_after + alpha * F(remaining_after)
    
    This encourages:
    - Small remainders (best-fit)
    - But penalizes extremely small remainders that are unlikely to be filled
    - Rewards moderate remainders that have high fill probability
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple fallback (pure best-fit)
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    feasible_remaining = remaining_after[feasible]
    
    # Best-fit component: -remaining_after (prefer smaller remainders)
    best_fit_score = -feasible_remaining
    
    # Regret component: Weibull CDF of the remainder
    # Higher CDF means higher probability a future item fits in this space
    fill_prob = _weibull_cdf(feasible_remaining)
    
    # Combined score: balance between best-fit and fill probability
    # alpha controls the weight of the regret term
    alpha = 0.3
    combined_score = best_fit_score + alpha * fill_prob
    
    scores[feasible] = combined_score
    
    return scores
# EVOLVE-BLOCK-END
