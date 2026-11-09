"""Stochastic inventory optimization for online bin packing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters for the item size distribution
# We'll use typical Weibull parameters that generate items in [0, 100]
# Common choices: shape k ~ 2-4, scale lambda ~ 20-30
# The hazard function h(x) = (k/lambda) * (x/lambda)^(k-1)
# The survival function S(x) = exp(-(x/lambda)^k)

# Precompute some useful constants
_WEIBULL_K = 3.0  # shape parameter
_WEIBULL_LAMBDA = 25.0  # scale parameter
_CAPACITY = 100.0

# Time guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit


def _weibull_hazard(x):
    """Compute Weibull hazard function h(x) = (k/lambda) * (x/lambda)^(k-1)."""
    x = np.clip(x, 1e-10, _CAPACITY)
    return (_WEIBULL_K / _WEIBULL_LAMBDA) * (x / _WEIBULL_LAMBDA) ** (_WEIBULL_K - 1)


def _weibull_survival(x):
    """Compute Weibull survival function S(x) = exp(-(x/lambda)^k)."""
    x = np.clip(x, 0, _CAPACITY)
    return np.exp(-((x / _WEIBULL_LAMBDA) ** _WEIBULL_K))


def _expected_future_waste(remaining):
    """
    Compute the expected future waste for a bin with given remaining capacity.
    
    The idea: if a bin has 'remaining' capacity left, the expected waste is
    the integral over possible next item sizes of the waste that item would
    cause, weighted by the probability of that item size.
    
    For a remaining capacity R, the expected waste from the next item that fits
    is E[max(0, R - X)] where X is the next item size, but only for X <= R.
    Items larger than R won't fit in this bin.
    
    More precisely, the value of preserving remaining capacity R is the expected
    reduction in total waste. If we use capacity R now (by placing an item),
    we avoid future waste. The marginal value of capacity is the hazard-weighted
    probability that a future item will fit in exactly that space.
    
    We approximate: value(R) = ∫₀ᴿ S(x) * h(x) * (R - x) dx
    This is the expected waste that would be incurred if we had to leave R unused.
    
    Actually, let's think differently. The "cost" of having remaining capacity R
    is that it represents potential waste. The expected waste contributed by this
    bin is E[min(R, next_item_that_fits)] but that's complex.
    
    Simpler model: The marginal value of one more unit of capacity in a bin with
    remaining R is the probability that a future item of size exactly R' (for R' 
    near R) would fit and use that space. This is related to the survival function.
    
    Let's use: waste_value(R) = ∫₀ᴿ S(x) dx
    This represents the expected "usable" capacity remaining, where S(x) is the
    probability that a future item is at least size x (so it could use space up to x).
    
    Actually, a cleaner interpretation: 
    - The expected number of future items that can fit in capacity R is related to 
      the integral of the survival function.
    - The waste is R minus the expected total size of items that will fit.
    
    Let's use a practical approximation:
    waste(R) = R * (1 - S(R)) + ∫₀ᴿ x * f(x) dx * (probability item fits)
    
    This is getting complex. Let's use a simpler but effective model:
    
    The "badness" of remaining capacity R is:
    - If R is very small (< min_item_size typically), it's pure waste
    - If R is moderate, it might fit one more item
    - If R is large, it can fit multiple items
    
    We'll model the expected waste as:
    waste(R) = R * (1 - S(R)) + R * S(R) * (1 - R/E[X|X<=R])
    
    Where E[X|X<=R] is the conditional expectation of item size given it fits.
    
    For Weibull, E[X|X<=R] can be approximated. But let's use a simpler proxy:
    
    The key insight: we want to minimize the "regret" of placing an item.
    If we place an item of size `item` in a bin with remaining `R`, the new remaining
    is R - item. The change in expected waste is:
    Δwaste = waste(R - item) - waste(R)
    
    We want to minimize this Δwaste (make it as negative as possible, or least positive).
    
    For computational efficiency, let's precompute a lookup table for waste(R).
    """
    # Use a piecewise polynomial approximation for speed
    # waste(R) ≈ R * (1 - S(R)) for the "unusable" portion
    # Plus a correction for the "partially usable" portion
    
    R = np.asarray(remaining, dtype=np.float64)
    
    # S(R) = probability that a random item is > R
    S_R = _weibull_survival(R)
    
    # The expected waste: 
    # - With probability S(R), no future item fits, so all R is waste
    # - With probability 1-S(R), some items fit, but not all of R is used
    # Approximate: waste ≈ R * S(R) + (1 - S(R)) * R * (1 - 0.5 * R / _WEIBULL_LAMBDA)
    # The 0.5 factor is a rough estimate of the fraction of R that gets used
    
    # Better: waste(R) ≈ R * S(R) + (1 - S(R)) * max(0, R - _WEIBULL_LAMBDA * (1 - S(R)))
    # This says: if items fit, they use approximately lambda * (1-S(R)) worth of space
    
    lambda_eff = _WEIBULL_LAMBDA * (1 - S_R)
    waste_from_fit = np.maximum(0, R - lambda_eff)
    
    waste = R * S_R + (1 - S_R) * waste_from_fit
    
    return waste


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing `item`. Higher score = better placement.
    
    Uses a stochastic inventory optimization model where the score reflects
    the expected future waste reduction from placing the item in each bin.
    """
    # Check time
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback to simple best-fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    # For invalid bins, assign a very low score
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
    
    rem_valid = remaining[valid]
    bins_valid = bins[valid]
    
    # Compute the expected waste before and after placement
    # Before: waste(bins_valid)
    # After: waste(rem_valid)
    # The change in waste: Δwaste = waste(rem_valid) - waste(bins_valid)
    # We want to minimize Δwaste, so score = -Δwaste
    
    waste_before = _expected_future_waste(bins_valid)
    waste_after = _expected_future_waste(rem_valid)
    
    delta_waste = waste_after - waste_before
    
    # Score is negative of delta waste (lower delta waste = higher score)
    scores[valid] = -delta_waste
    
    return scores
# EVOLVE-BLOCK-END
