import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

# Weibull CDF for small items (shape k=1.5, scale lambda=50)
# P(X <= x) = 1 - exp(-(x/lambda)^k)
# We want to score based on how "likely" a remaining capacity is to be filled by a future small item.
# If remaining is small, it's likely to be filled. If remaining is large, it's less likely.
# So we want to prefer bins where remaining is small (high probability of being filled).
# Score should be high for small remaining.
# Let's use the CDF directly as a proxy for "usefulness" of the remaining space.
# But we also need to account for the fact that we are placing the current item.
# The current program uses:
# Large: Worst Fit (max remaining)
# Small: Best Fit with penalty (min remaining, with sliver avoidance)

# The inspiration suggests using Weibull CDF to penalize large remainders more heavily for small items.
# This aligns with "Best Fit" but with a non-linear scoring.
# Let's try scoring small items using the Weibull CDF of the remaining capacity.
# CDF(x) = 1 - exp(-(x/lambda)^k)
# For small x, CDF is small. For large x, CDF is close to 1.
# We want to prefer small remaining. So we want high score for small remaining.
# So we can use Score = -CDF(remaining) or Score = exp(-(remaining/lambda)^k).
# Let's try Score = exp(-(remaining/lambda)^k) for small items.
# This gives high score for small remaining, low score for large remaining.
# This is a smooth, differentiable function.

def _weibull_score(remaining, k=1.5, lam=50.0):
    # remaining is an array of valid remaining capacities
    # We want to score such that smaller remaining is better (higher score)
    # exp(-(x/lam)^k) is decreasing in x
    return np.exp(-((remaining / lam) ** k))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    Strategy: Hybrid Best/Worst Fit based on item size relative to capacity.
    - Large items (> 50): Worst Fit (leave max space to avoid fragmentation for other large items)
    - Small items (<= 50): Weibull CDF-based scoring (prefer small remainders, smooth penalty for large)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        # Fallback: simple best fit
        scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid):
        return scores

    valid_remaining = remaining[valid]
    valid_indices = np.where(valid)[0]

    if item > 50.0:
        # Worst Fit: maximize remaining capacity
        scores[valid_indices] = valid_remaining
    else:
        # Weibull-based scoring for small items
        # Prefer bins where remaining capacity is small (likely to be filled by future small items)
        # Use exp(-(x/lam)^k) which is high for small x
        # This smoothly penalizes large remainders without a hard sliver threshold
        scores[valid_indices] = _weibull_score(valid_remaining, k=1.5, lam=50.0)

    return scores
