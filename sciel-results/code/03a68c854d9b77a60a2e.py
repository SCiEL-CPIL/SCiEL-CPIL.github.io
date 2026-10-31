"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Constants
_CAPACITY = 100.0
_DEADLINE = 150.0
_START_TIME = time.time()

# Pre-computed constants for the score function
# We use a combination of best-fit with a "future-aware" penalty
# The idea is to avoid creating gaps that are too small to be useful
# but also not too large (which wastes capacity).

def _compute_score(item: float, remaining: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """
    Compute scores for feasible bins.
    
    Strategy: A structured heuristic that considers:
    1. Best-fit base (minimize remaining gap)
    2. Gap utility: penalize gaps that are too small (< item/2) or in a "dead zone"
    3. Balance: slight preference for bins that keep the overall distribution balanced
    
    This is a structural departure from pure sigmoid blending.
    """
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
    
    gaps = remaining[feasible]
    n_feasible = np.sum(feasible)
    
    # --- Component 1: Best-Fit Base ---
    # Normalize gaps to [0, 1] for consistent scaling
    # We want to prefer smaller gaps, so base_score = -gap
    base = -gaps
    
    # --- Component 2: Gap Utility Penalty ---
    # Gaps that are too small (< 50% of current item) are likely useless
    # Gaps in the "dead zone" (e.g., 50-100) are also suboptimal
    # We want to encourage gaps that are either very small (tight fit) 
    # or large enough to fit future items
    
    # Small gap penalty: if gap < item * 0.4, it's probably useless
    small_gap_threshold = item * 0.4
    small_gap_penalty = np.where(gaps < small_gap_threshold, 
                                  (small_gap_threshold - gaps) / small_gap_threshold * 5.0, 
                                  0.0)
    
    # Large gap penalty: if gap > 0.8 * capacity, we're wasting a lot
    # But this is less critical than small gaps
    large_gap_threshold = _CAPACITY * 0.8
    large_gap_penalty = np.where(gaps > large_gap_threshold,
                                  (gaps - large_gap_threshold) / _CAPACITY * 2.0,
                                  0.0)
    
    gap_penalty = small_gap_penalty + large_gap_penalty
    
    # --- Component 3: Balance Term ---
    # To avoid all items going to the same few bins, we add a small 
    # preference based on how "full" the bin is relative to others
    # This is a soft load-balancing term
    
    # Compute the mean remaining capacity of feasible bins
    if n_feasible > 1:
        mean_gap = np.mean(gaps)
        # Prefer bins closer to the mean (balanced)
        # Use a quadratic penalty for deviation from mean
        balance_penalty = 0.1 * (gaps - mean_gap) ** 2 / (_CAPACITY * _CAPACITY)
    else:
        balance_penalty = 0.0
    
    # --- Combine ---
    # Weight the components
    w_base = 1.0
    w_gap = 1.5
    w_balance = 1.0
    
    final_scores = w_base * base - w_gap * gap_penalty - w_balance * balance_penalty
    
    scores[feasible] = final_scores
    return scores


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Structured heuristic with best-fit base, gap utility, and balance.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 2.0:
        # Emergency fallback: simple best-fit
        remaining = bins - item
        feasible = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(feasible):
            scores[feasible] = -remaining[feasible]
        return scores
    
    remaining = bins - item
    feasible = remaining >= 0
    
    return _compute_score(item, remaining, feasible)
# EVOLVE-BLOCK-END
