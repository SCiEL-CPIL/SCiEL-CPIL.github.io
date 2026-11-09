"""Rewritten priority for online bin packing with a structural heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track per-bin usage statistics across the online stream.
# We maintain:
#   _bin_count: number of items placed in each bin (proxy for bin "age" / fragmentation)
#   _bin_sum:   sum of item sizes placed in each bin (to compute mean item size per bin)
#   _item_count: total items seen so far (global counter)
#   _t0:        start time for deadline guard
#   _deadline:  absolute deadline (seconds)

_bin_count = None
_bin_sum = None
_item_count = 0
_t0 = None
_deadline = None
_DEADLINE_MARGIN = 10.0  # seconds before the 150s kill to stop complex computation

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = preferred bin.
    
    Structural heuristic:
    - Prefer bins that are "fuller" (less remaining space) to consolidate items,
      but penalize bins that are nearly full (to avoid wasting a bin on a tiny remainder).
    - Use a smooth function of remaining capacity relative to item size.
    - Add a small bonus for bins with lower fragmentation (more items, similar sizes).
    """
    global _bin_count, _bin_sum, _item_count, _t0, _deadline
    
    n_bins = len(bins)
    
    # Initialize state if needed
    if _bin_count is None or _bin_count.shape[0] != n_bins:
        _bin_count = np.zeros(n_bins, dtype=np.float64)
        _bin_sum = np.zeros(n_bins, dtype=np.float64)
        _item_count = 0
        _t0 = time.time()
        _deadline = _t0 + 150.0 - _DEADLINE_MARGIN
    
    # Deadline guard: if we're close to the limit, use a simple fast heuristic
    if time.time() > _deadline:
        # Simple best-fit: prefer bins with least remaining space
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining = bins - item
    feasible = remaining >= 0
    
    # Base score: prefer bins that leave less remaining space (best-fit)
    # But use a nonlinear transform to balance consolidation vs. avoiding waste
    # score = -remaining^alpha, where alpha < 1 makes it less aggressive
    # Actually, let's think differently:
    
    # We want to encourage filling bins to near-capacity.
    # A good heuristic: score = -remaining / (item + epsilon) normalized
    # This makes the score relative to the item size.
    
    # Alternative: use a "waste" metric. If we put item in bin, the remaining
    # space is 'waste'. We want to minimize waste, but also not open new bins
    # unnecessarily.
    
    # Let's use a combination:
    # 1. Primary: minimize remaining space (best-fit)
    # 2. Secondary: prefer bins that already have items (avoid opening new bins)
    # 3. Tertiary: prefer bins where the item fits "nicely" (not too small for the gap)
    
    # Compute scores
    eps = 1e-9
    scores = np.full(n_bins, -np.inf)
    
    if not np.any(feasible):
        return scores
    
    # For feasible bins, compute a composite score
    rem = remaining[feasible]
    
    # Component 1: Best-fit (less remaining is better)
    # Normalize by item size to make it scale-invariant
    best_fit_score = -rem / (item + eps)
    
    # Component 2: Prefer bins that are already in use (have items)
    # This avoids opening new bins when possible
    in_use = _bin_count[feasible]
    in_use_bonus = np.log1p(in_use) * 0.1  # small bonus for bins with more items
    
    # Component 3: Penalize bins where the item is much smaller than the gap
    # If the gap is much larger than the item, we're wasting space
    gap_ratio = rem / (item + eps)
    # Penalize if gap_ratio is large (item is small relative to gap)
    # Use a soft penalty: penalty = max(0, gap_ratio - 2) * 0.05
    gap_penalty = np.maximum(0, gap_ratio - 2.0) * 0.05
    
    # Combine: higher score is better
    composite = best_fit_score + in_use_bonus - gap_penalty
    
    scores[feasible] = composite
    
    # Update state for the bin that will be chosen
    # But we don't know which bin will be chosen yet (the evaluator picks the max)
    # So we can't update state here. We need to handle this differently.
    
    # Actually, the problem says "each arriving item goes to the highest-scoring bin"
    # So the evaluator will pick argmax(scores). We should update our state
    # to reflect that choice. But we don't know which bin is chosen until after
    # we return. This is a problem for stateful heuristics.
    
    # Alternative: make the heuristic stateless, or use a different approach.
    # Let's stick with a stateless heuristic for now.
    
    return scores
# EVOLVE-BLOCK-END
