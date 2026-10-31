"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    Lower raw evaluation value (mean percent excess over L1 lower bound) is better.
    """
    # Deadline guard: if we're getting close to the limit, fall back to a fast heuristic
    # We track a global start time to enforce a soft deadline
    if not hasattr(priority, "_start_time"):
        priority._start_time = time.time()
    elapsed = time.time() - priority._start_time
    if elapsed > 135.0:  # Safety margin before 150s kill
        # Fast fallback: best-fit-decreasing style
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Filter bins with enough capacity
    remaining = bins - item
    valid = remaining >= 0

    if not np.any(valid):
        # No valid bin; return all -inf (shouldn't happen in valid evaluation)
        return np.full(len(bins), -np.inf)

    # Strategy: Weibull-informed best-fit.
    # The insight from the measured improvement: using a Weibull CDF-based score
    # that penalizes large remainders more heavily aligns better with Weibull sequences.
    #
    # For online bin packing with Weibull-distributed items, the optimal policy
    # favors placing items in bins where the remaining space is most likely to be
    # filled by future items. This is related to the "expected future fill" of the remainder.
    #
    # A smooth surrogate approach: instead of hard best-fit (minimize remainder),
    # use a soft-min via log-sum-exp to create a smooth scoring function,
    # combined with a Weibull-shaped penalty on the remainder.
    #
    # The Weibull CDF: F(x) = 1 - exp(-(x/scale)^shape)
    # For Weibull items with shape ~1.5-2.5 (typical for bin packing benchmarks),
    # small items are more common. So a remainder that is small is more likely
    # to be filled. We want to maximize the probability that the remainder gets filled.
    #
    # Score = -penalty(remainder) where penalty increases with remainder,
    # but shaped by Weibull distribution properties.
    #
    # Key idea: the "value" of a remainder r is the expected number of future items
    # that can fit in it. For Weibull-distributed items, this is approximately
    # related to the CDF evaluated at r.
    #
    # We use: score = -(remainder / scale)^shape * exp(-remainder/scale) + log(1 + 1/remainder)
    # This creates a non-linear penalty that's steeper for medium remainders and
    # flatter for very small or very large remainders.
    #
    # Actually, let's think more carefully. The measured improvement used a
    # Weibull CDF-based score. Let me design this more carefully.
    #
    # For a bin with remaining capacity r after placing the current item:
    # - If r is very small, it's "wasted" but the bin is nearly full (good packing)
    # - If r is medium, it might be fillable by one more item (potentially good)
    # - If r is large, it's likely to be wasted (bad packing)
    #
    # The L1 lower bound is sum(items) / capacity. We want to minimize the number
    # of bins used, which means maximizing the fill rate per bin.
    #
    # A good heuristic: minimize the "effective waste" which accounts for the
    # probability that the remainder will be filled. For Weibull items with
    # shape k and scale lambda, the probability that a future item fits in
    # remainder r is P(X <= r) = 1 - exp(-(r/lambda)^k).
    #
    # So the expected future fill of remainder r is roughly:
    #   E[fill] = sum over possible numbers of items that fit
    # For simplicity, approximate as: the remainder is "valuable" proportional to
    # P(item fits in r) * E[item size | item <= r]
    #
    # A simpler approach that worked: score = -f(r) where f is a Weibull-shaped
    # function that's minimized at r=0 and increases with r.
    #
    # Let's use: f(r) = r * (1 + exp(-(r - r0) / tau))
    # where r0 is a threshold and tau controls the smoothness.
    # This is a smooth step-like function: for r < r0, f ~ r; for r > r0, f ~ 2r.
    # This penalizes large remainders more heavily.
    #
    # Better: use the actual Weibull CDF to weight the penalty.
    # Assume typical Weibull parameters for the benchmark: shape k=2, scale lambda=30
    # (items in [0, 100] with capacity 100, 1000 items, mean ~30-40)
    #
    # Score for a bin with remainder r:
    #   base_score = -r  (best-fit baseline)
    #   weibull_bonus = log(1 + P_fill(r))  where P_fill(r) = 1 - exp(-(r/lambda)^k)
    #   total_score = base_score + weibull_bonus * weight
    #
    # This gives a small bonus to bins with remainders that are likely to be filled.

    valid_indices = np.where(valid)[0]
    r = remaining[valid_indices]

    # Weibull parameters tuned for typical bin packing benchmarks
    # Items are Weibull with capacity 100, 1000 items
    # Typical shape k in [1.5, 3.0], scale lambda in [20, 50]
    # Use k=2, lambda=35 as a reasonable default
    k = 2.0
    lam = 35.0

    # P(fill) = probability a future item fits in remainder r
    # For Weibull: P(X <= r) = 1 - exp(-(r/lam)^k)
    # Handle r=0 carefully
    p_fill = 1.0 - np.exp(-np.power(r / lam, k))

    # The "value" of the remainder: how much of it is likely to be used
    # Expected fill contribution ~ p_fill * min(r, lam) (approximate)
    # Use a smooth approximation: value = r * p_fill
    # This is 0 when r=0, and approaches r as r -> inf (but p_fill -> 1)

    # Score: we want to MINIMIZE waste, so we maximize a score where higher is better
    # The "waste" is approximately r * (1 - p_fill) + small_penalty * r * p_fill
    # i.e., the part of the remainder that won't be filled is true waste
    # The part that will be filled is less of a concern

    # waste = r * (1 - p_fill)
    # score = -waste = -r * (1 - p_fill) = -r * exp(-(r/lam)^k)

    # This is a smooth, non-linear penalty:
    # - For small r: exp(-(r/lam)^k) ~ 1, so score ~ -r (like best-fit)
    # - For large r: exp(-(r/lam)^k) ~ 0, so score ~ 0 (large remainders not penalized as much)
    # Wait, that's backwards. Large remainders should be penalized MORE.

    # Let me reconsider. The issue is:
    # - Best-fit minimizes r, which is good for reducing bins
    # - But if r is large, it means the bin is underutilized, which is bad
    # - If r is small, the bin is well-utilized, which is good
    #
    # So we want score to be HIGH (good) when r is small, LOW (bad) when r is large.
    # Best-fit: score = -r, which does exactly this.
    #
    # The Weibull insight: not all remainders are equally "bad."
    # A remainder of 10 is better than a remainder of 50 because:
    # 1. It's closer to zero (less waste)
    # 2. It's more likely to be filled by a future small item
    #
    # The penalty should be non-linear: steeper for larger remainders.
    # Using the Weibull CDF: penalty(r) = r * (1 - F(r)) + c * F(r) * g(r)
    # where F(r) = 1 - exp(-(r/lam)^k)
    #
    # Simpler: penalty(r) = r / (1 + alpha * F(r))
    # This reduces the effective penalty for remainders that are likely to be filled.
    # But we want to PENALIZE large remainders more, not less.

    # Let me try a different approach:
    # The "effective waste" of a remainder r is:
    #   - If r will be filled: waste = 0 (the space gets used)
    #   - If r won't be filled: waste = r
    # Expected waste = r * P(not filled) = r * exp(-(r/lam)^k)
    #
    # But this UNDER-penalizes large remainders because P(not filled) -> 0.
    # That's wrong: a large remainder that won't be filled is MORE waste, not less.

    # Correct model:
    # The number of bins used depends on how well items pack.
    # Best-fit works because it minimizes fragmentation.
    # A remainder r creates "fragmentation" that can only be filled by items <= r.
    # The "cost" of this fragmentation is:
    #   - If no future item fits: cost = r (wasted space)
    #   - If a future item fits: cost = 0 (space gets used)
    # But the PROBABILITY that no future item fits decreases with r (larger r means
    # more items can fit).
    #
    # Actually, for minimizing bins, what matters is:
    #   - Total capacity used = sum of item sizes
    #   - Number of bins = ceil(total_capacity_used / capacity) at minimum (L1 bound)
    #   - Excess bins come from fragmentation: when a bin can't accept the next item
    #     despite having remaining space
    #
    # So the key is to minimize the "dead space" - remaining capacity that can't be
    # filled by any future item. For a remainder r, the dead space is r if no future
    # item is <= r, and 0 if some future item is <= r (approximately).
    #
    # Expected dead space for remainder r:
    #   E[dead] = r * P(no future item <= r)
    # For n remaining items, P(no future item <= r) = (1 - F(r))^n
    # But n varies, so use a geometric approximation or just (1 - F(r))^k for some k.
    #
    # This gives: E[dead] = r * (exp(-(r/lam)^k))^n = r * exp(-n * (r/lam)^k)
    # For large n, this is very small for any r > 0, which means all remainders
    # are likely to be filled. That's not quite right either.

    # Let me just go with a practical approach that the measured improvement used:
    # A Weibull CDF-based score that penalizes large remainders more heavily.
    #
    # score = -r - beta * r^2 / (r^2 + gamma^2)
    # This adds a quadratic penalty that saturates, making large remainders more costly.
    #
    # Or even simpler, based on the measured improvement description:
    # "penalizes large remainders more heavily (or differently) than linear"
    #
    # Use: score = -r * (1 + r / gamma)
    # This is a quadratic penalty: score = -r - r^2/gamma
    # For small r: ~ -r (like best-fit)
    # For large r: ~ -r^2/gamma (much more penalized)

    # Let's use a smooth, Weibull-inspired penalty:
    # penalty(r) = r * (1 + (r/lam)^k)
    # score = -penalty(r)

    # This is smooth, non-linear, and penalizes large remainders more heavily.
    # For r << lam: penalty ~ r (best-fit)
    # For r >> lam: penalty ~ r * (r/lam)^k = r^(k+1) / lam^k (super-linear penalty)

    penalty = r * (1.0 + np.power(r / lam, k))
    scores_valid = -penalty

    # Construct full score array
    scores = np.full(len(bins), -np.inf)
    scores[valid_indices] = scores_valid

    return scores
# EVOLVE-BLOCK-END
