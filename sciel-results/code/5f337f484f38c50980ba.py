"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Enhanced best-fit with consolidation bias.
    We want to pack items as tightly as possible to minimize the number of bins used,
    which relates to minimizing excess over the L1 lower bound.
    
    L1 lower bound is sum(items) / capacity.
    To minimize excess, we want to minimize actual_bins_used, which means packing tightly.
    
    Key insight: Pure best-fit (minimize remaining after placement) is a strong baseline.
    However, we can enhance it by:
    1. Preferring bins that are already partially filled (consolidation)
    2. Avoiding bins that are almost empty (to save them for larger future items)
    
    The score combines:
    - Primary: minimize remaining after placement (best-fit)
    - Secondary: prefer bins with more used capacity (consolidation)
    
    Since bins[i] represents remaining capacity, used capacity = capacity - bins[i].
    But we don't know capacity directly. However, we can use the fact that
    bins with smaller remaining values are more "used up" (if they started at the same capacity).
    
    Actually, all bins start at the same capacity (100 in the task). So:
    - used[i] = capacity - bins[i]
    - remaining_after = bins[i] - item
    
    For best-fit: minimize remaining_after = bins[i] - item
    For consolidation: maximize used[i] = -bins[i] (since capacity is constant)
    
    So both objectives point in the same direction: prefer smaller bins[i] values.
    
    But we need to be careful: if a bin has very little remaining, placing a large item
    might not fit. We already filter for feasible bins (remaining >= 0).
    
    Let's use a weighted combination:
    score = -remaining_after - alpha * bins[i]
    
    Wait, that's redundant. Let me think again.
    
    Actually, the key insight is that we want to consolidate items into fewer bins.
    This means we should prefer bins that are already in use over empty bins.
    
    An empty bin has bins[i] = capacity (e.g., 100).
    A partially used bin has bins[i] < capacity.
    
    If we just use best-fit (-remaining_after), we might place an item in an empty bin
    if it's the only one that fits well, when we could have used a partially filled bin.
    
    Better approach: Add a bonus for bins that are not empty.
    
    Let's define:
    - remaining_after = bins[i] - item (must be >= 0)
    - used_fraction = 1 - bins[i] / capacity (but we don't know capacity)
    
    Alternative: Use the sign/magnitude of bins[i] as a proxy for usage.
    Since all bins start at the same capacity, smaller bins[i] means more used.
    
    Score = -remaining_after + beta * (-bins[i]) = -remaining_after - beta * bins[i]
    
    This penalizes bins with large remaining capacity (empty bins) and rewards bins
    with small remaining capacity (full bins).
    
    But wait, -remaining_after = -(bins[i] - item) = -bins[i] + item
    So score = -bins[i] + item - beta * bins[i] = item - (1 + beta) * bins[i]
    
    Since item is constant for all bins, this is equivalent to minimizing bins[i],
    which is just best-fit! So the consolidation bias is already captured by best-fit.
    
    The issue is that best-fit might be too aggressive. Let's try a different angle.
    
    What if we consider the "waste" created by placing the item?
    Waste = remaining_after = bins[i] - item
    
    We want to minimize waste, but also consider the opportunity cost.
    If we use an empty bin, we "waste" the opportunity to use it for a larger item later.
    
    Heuristic: Penalize empty bins more heavily.
    
    Let's define:
    - is_empty = bins[i] >= capacity * threshold (e.g., 0.95)
    - If empty, add a penalty
    
    But we don't know capacity. However, we can estimate it from the maximum bins value.
    
    Actually, let's try a simpler enhancement:
    score = -remaining_after + epsilon * used_estimate
    
    where used_estimate is a proxy for how much of the bin is already used.
    
    Since we don't know capacity, let's use:
    used_estimate = -bins[i]  (smaller bins[i] means more used)
    
    score = -remaining_after + epsilon * (-bins[i])
    = -(bins[i] - item) - epsilon * bins[i]
    = -bins[i] + item - epsilon * bins[i]
    = item - (1 + epsilon) * bins[i]
    
    Again, this is just a scaled version of best-fit.
    
    Let me try a completely different approach: What if we prefer bins where
    the item fills a "nice" fraction of the remaining space?
    
    Or: What if we use a two-tier scoring?
    1. First, filter to bins where remaining_after is "small" (good fit)
    2. Among those, prefer bins that are more used
    
    Let's implement:
    - Compute remaining_after = bins[i] - item
    - Filter feasible bins
    - Among feasible, compute a composite score
    
    Composite score idea:
    - Primary: -remaining_after (best-fit)
    - Secondary: -bins[i] (prefer more used bins)
    
    But as shown, these are linearly dependent. So we need a non-linear combination.
    
    How about: score = -remaining_after * (1 + gamma * used_fraction)
    
    where used_fraction is estimated as 1 - bins[i] / max_bins
    
    This gives extra weight to best-fit when the bin is already used.
    
    Let's implement this.
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    # Estimate used fraction: 1 - bins[i] / max_bins
    # max_bins is an estimate of capacity
    max_bins = np.max(bins) if len(bins) > 0 else 1.0
    if max_bins <= 0:
        max_bins = 1.0
    
    used_fraction = 1.0 - bins / max_bins
    # Clip to [0, 1] for safety
    used_fraction = np.clip(used_fraction, 0.0, 1.0)
    
    # Composite score: -remaining_after * (1 + gamma * used_fraction)
    gamma = 2.0  # Tunable parameter
    scores = np.full(len(bins), -np.inf)
    
    feasible_remaining = remaining_after[feasible]
    feasible_used = used_fraction[feasible]
    
    # Base score: -remaining_after
    base_score = -feasible_remaining
    
    # Multiplier: 1 + gamma * used_fraction
    multiplier = 1.0 + gamma * feasible_used
    
    scores[feasible] = base_score * multiplier
    
    return scores
# EVOLVE-BLOCK-END
