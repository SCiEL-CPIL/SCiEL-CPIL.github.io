"""Improved program: Adaptive best/worst-fit hybrid with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level time tracking for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing the given item.
    bins: array of remaining capacities.
    Returns: array of scores, higher is better.
    Only bins with enough capacity are scored; others get -inf.
    
    Strategy: Adaptive hybrid of best-fit and worst-fit.
    - When bins are densely packed (high average fill), use best-fit (minimize gap).
    - When bins are sparse (low average fill), blend in worst-fit to preserve flexibility.
    """
    # Deadline guard: if we're close to timeout, use simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        mask = bins >= item
        if not np.any(mask):
            return -np.inf * np.ones_like(bins)
        scores = -np.inf * np.ones_like(bins)
        gaps = bins[mask] - item
        scores[mask] = -gaps  # simple best-fit
        return scores
    
    remaining = bins
    mask = remaining >= item
    
    if not np.any(mask):
        return -np.inf * np.ones_like(remaining)
    
    valid_remaining = remaining[mask]
    gaps = valid_remaining - item
    
    # Compute average fill level across ALL bins (not just valid ones)
    # This tells us how "packed" the system is
    avg_fill = 1.0 - np.mean(remaining) / 100.0
    
    # Also consider the distribution of gaps: if gaps are tightly clustered,
    # best-fit is safer. If gaps are spread out, worst-fit preserves options.
    gap_spread = np.std(gaps) if len(gaps) > 1 else 0.0
    gap_mean = np.mean(gaps) if len(gaps) > 0 else 0.0
    
    # Adaptive weight: blend between best-fit (weight on -gap) and worst-fit (weight on +gap)
    # When avg_fill is high (>0.7), we're in dense packing -> favor best-fit
    # When avg_fill is low (<0.3), we're in sparse packing -> favor worst-fit
    # gap_spread also modulates: high spread means more uncertainty -> more worst-fit
    
    # Base: best-fit score = -gap (lower gap is better)
    # Worst-fit component: +gap (higher gap preserves more space for future)
    
    # Weight for worst-fit component: increases as fill decreases and spread increases
    # Normalize spread relative to mean to get a relative measure
    spread_ratio = gap_spread / (gap_mean + 1e-9) if gap_mean > 0 else 0.0
    
    # worst_fit_weight in [0, 0.5]
    fill_component = max(0.0, 0.5 - avg_fill)  # 0.5 when empty, 0 when 50%+ full
    spread_component = min(0.5, spread_ratio * 0.3)
    worst_fit_weight = fill_component + spread_component
    worst_fit_weight = min(worst_fit_weight, 0.5)  # cap at 0.5
    
    # Combined score: -gap * (1 - 2*w) + gap * w = -gap + 2*w*gap = gap*(2*w - 1)
    # When w=0: score = -gap (best fit)
    # When w=0.5: score = 0 (neutral)
    # Let's use: score = -gap * (1 - worst_fit_weight) + gap * worst_fit_weight
    #           = gap * (2 * worst_fit_weight - 1)
    
    if worst_fit_weight < 0.25:
        # Mostly best-fit
        scores_valid = -gaps
    elif worst_fit_weight > 0.4:
        # More balanced, slight preference for larger gaps
        scores_valid = gaps * (2.0 * worst_fit_weight - 1.0)
    else:
        # Transitional: use a smooth blend
        w = worst_fit_weight
        scores_valid = -gaps * (1.0 - w) + gaps * w
    
    # Create full score array
    scores = -np.inf * np.ones_like(remaining)
    scores[mask] = scores_valid
    
    return scores

# EVOLVE-BLOCK-END
