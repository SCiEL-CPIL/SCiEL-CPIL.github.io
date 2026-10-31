"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

# Weibull distribution parameters for the generated sequences
# Assuming a standard Weibull distribution with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and a median of ~19, which fits well into capacity 100.
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
k = 2.0
lambda_scale = 25.0
expected_next_item = lambda_scale * scipy.special.gamma(1 + 1/k)

# Golden ratio target for "stable" remainder
# The idea is that a remainder of ~0.618 * Capacity is a "good" state because
# it's large enough to likely fit another item, but small enough that we haven't
# wasted too much space in a bin that could have been filled more tightly.
# Capacity is assumed to be 100.
GOLDEN_TARGET = 0.618 * 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback: Best Fit (minimize remaining)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Golden Ratio Attractor with Probabilistic Fragmentation Penalty
    #
    # 1. Calculate the gap for each valid bin: g = bins - item.
    # 2. Compute a "Golden Ratio" score: -|g - GOLDEN_TARGET|
    #    This creates an attractor at g = GOLDEN_TARGET.
    #    It discourages extremely small gaps (fragmentation) and extremely large gaps (waste),
    #    pushing the system towards a stable state where bins have "useful" remainders.
    #
    # 3. Compute a "Probabilistic Fragmentation" penalty.
    #    If the gap is smaller than the expected next item, it's likely dead space.
    #    We apply a smooth penalty to discourage creating such gaps.
    #    This is inspired by the successful "probabilistic gap penalty" approach.
    #
    # 4. Combine the two scores.
    #    The Golden Ratio score provides a global structure to the bin states.
    #    The Probabilistic Penalty provides a local guard against immediate fragmentation.
    
    valid_remaining = bins[valid]
    gap = valid_remaining - item
    
    # Golden Ratio Attractor Score
    # We want to maximize the score, so we use negative distance.
    golden_score = -np.abs(gap - GOLDEN_TARGET)
    
    # Probabilistic Fragmentation Penalty
    # Smooth penalty if gap < expected_next_item
    # We use a smooth step function to avoid discontinuities.
    # Let x = (gap - expected_next_item) / sigma
    # penalty = -exp(x) if x < 0, else 0? 
    # Actually, let's use a smooth max(0, expected_next_item - gap)
    # A common smooth approximation for max(0, z) is softplus(z) - softplus(0) = softplus(z)
    # But we want to penalize small gaps.
    # Let's use: penalty = -exp(-gap / sigma) * (gap < expected_next_item)
    # This is not smooth.
    
    # Let's use the multiplier approach from Inspiration 1, which was effective.
    # multiplier = 1 + exp(-(gap - expected_next_item)/sigma)
    # This multiplier is large when gap is small, and ~1 when gap is large.
    # We want to penalize small gaps, so we subtract a term proportional to the multiplier?
    # Or we can just use the multiplier to scale the "cost" of the gap.
    
    # Let's try a different combination.
    # Base score: Golden Ratio Attractor
    # Penalty: Quadratic penalty for gaps that are too small.
    # penalty = - (max(0, expected_next_item - gap)) ** 2
    
    # Let's use a smooth penalty:
    # p = expected_next_item - gap
    # if p > 0, penalty = -p**2
    # Smoothly: penalty = -softplus(p)**2 ? No, softplus(p) is ~p for large p, ~exp(p) for small p.
    
    # Let's stick to the Inspiration 1's successful multiplier idea but apply it to the Golden Score?
    # Or add it as a separate term.
    
    # Let's try:
    # score = golden_score - penalty
    # where penalty is high when gap is small.
    
    sigma = 5.0
    # Smooth penalty for gap < expected_next_item
    # We want a function that is 0 when gap >> expected_next_item and large when gap << expected_next_item.
    # Let's use: penalty = exp(-(gap - expected_next_item)/sigma)
    # This is 1 when gap = expected_next_item, and large when gap is small.
    # This seems good.
    
    penalty = np.exp(-(gap - expected_next_item) / sigma)
    
    # Combine
    scores = golden_score - penalty
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores
    
    return scores_full
# EVOLVE-BLOCK-END
