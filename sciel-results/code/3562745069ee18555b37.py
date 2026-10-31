"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for estimating fill probability of remainders
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
    
    Strategy: Adaptive Best-Fit with Fragmentation Penalty.
    
    Core idea: We want to minimize the "dead space" created by placing an item.
    Dead space is remaining capacity that is too small to fit a typical future item.
    
    For a bin with remaining capacity R after placing the item:
    1. If R is very small (< threshold), it's likely dead space -> penalize.
    2. If R is moderate, it might fit a future item -> less penalty.
    3. If R is large, it's flexible -> least penalty (but we still prefer best-fit 
       to avoid wasting large bins on small items).
    
    We use a composite score:
    - Primary: Best-fit (minimize remaining_after)
    - Secondary: Fragmentation penalty based on P(future_item > remaining_after)
      which is the probability that the remaining space becomes dead.
    
    Score = -remaining_after - beta * (1 - F(remaining_after))
    where F is the Weibull CDF.
    
    This means:
    - Smaller remainders are preferred (best-fit)
    - But remainders that are "unfillable" (low CDF) get an extra penalty
    - Remainders that are "fillable" (high CDF) get less penalty
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
    
    # Fragmentation penalty: probability that remaining space is "dead"
    # i.e., P(future_item > remaining_after) = 1 - F(remaining_after)
    # where F is the Weibull CDF
    # High fragmentation penalty means the remaining space is likely unusable
    fill_prob = _weibull_cdf(feasible_remaining)
    frag_penalty = 1.0 - fill_prob  # Probability of dead space
    
    # Adaptive beta: 
    # - If item is large relative to capacity, we should be more aggressive with best-fit
    #   because opening new bins is expensive.
    # - If item is small, we can afford to be more selective.
    # 
    # Also, the penalty should scale with the magnitude of the remaining space.
    # A dead space of 5 units is worse than a dead space of 1 unit.
    # So we weight the fragmentation penalty by the remaining space itself.
    
    # Normalize remaining space to [0, 1] relative to capacity (100)
    capacity = 100.0
    normalized_remaining = feasible_remaining / capacity
    
    # Beta adapts based on item size and remaining space distribution
    # For large items, we want stronger best-fit (lower beta for frag penalty)
    # For small items, we can afford more selectivity (higher beta)
    item_ratio = item / capacity
    # Beta ranges from 0.1 (large items) to 0.5 (small items)
    beta = 0.1 + 0.4 * (1.0 - item_ratio)
    
    # Weighted fragmentation penalty: scale by normalized remaining space
    # This ensures that dead space in large bins is penalized more
    weighted_frag = beta * frag_penalty * normalized_remaining
    
    # Combined score
    combined_score = best_fit_score - weighted_frag
    
    scores[feasible] = combined_score
    
    return scores
# EVOLVE-BLOCK-END
