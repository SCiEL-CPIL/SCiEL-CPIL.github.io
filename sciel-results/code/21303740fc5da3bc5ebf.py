"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Phase-adaptive bin packing: transitions from Worst-Fit (early) to Best-Fit (late)
    based on the average remaining capacity across all bins.
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value (negative infinity) for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # If no bin can fit the item, return all low scores
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # Phase detection: use mean remaining capacity as a proxy for packing progress
    # Early phase: mean_remaining is high -> prefer Worst-Fit (larger remaining)
    # Late phase: mean_remaining is low -> prefer Best-Fit (smaller remaining)
    mean_remaining = np.mean(bins)
    
    # Normalize phase to [0, 1]: 0 = very early, 1 = very late
    # With capacity 100, early mean ~100, late mean ~10-20
    # Use a smooth transition centered around mean_remaining = 50
    phase = 1.0 / (1.0 + np.exp(0.15 * (mean_remaining - 50.0)))
    # phase close to 1 means late (prefer Best-Fit), close to 0 means early (prefer Worst-Fit)
    
    # Compute two candidate scores:
    # Best-Fit score: prefer bins with smallest remaining (most filled)
    best_fit_score = -valid_remaining
    
    # Worst-Fit score: prefer bins with largest remaining (least filled)
    worst_fit_score = valid_remaining
    
    # Blend: weight between Best-Fit and Worst-Fit based on phase
    # In early phase (phase~0), weight toward Worst-Fit
    # In late phase (phase~1), weight toward Best-Fit
    # Use a convex combination
    wf_weight = 1.0 - phase  # weight for Worst-Fit
    bf_weight = phase         # weight for Best-Fit
    
    # Normalize scores to comparable scale before blending
    # Best-Fit scores are negative (smaller remaining = less negative = better)
    # Worst-Fit scores are positive (larger remaining = better)
    # We want higher total score = better choice
    
    # For a clean blend, we can use:
    # score = bf_weight * best_fit_score + wf_weight * worst_fit_score
    # But these have different scales. Let's normalize each to [0,1] range within valid bins
    
    min_rem = np.min(valid_remaining)
    max_rem = np.max(valid_remaining)
    rem_range = max_rem - min_rem + 1e-9
    
    # Normalized Best-Fit: higher when remaining is smaller
    norm_bf = (max_rem - valid_remaining) / rem_range  # [0,1], 1 when remaining=min
    
    # Normalized Worst-Fit: higher when remaining is larger  
    norm_wf = (valid_remaining - min_rem) / rem_range   # [0,1], 1 when remaining=max
    
    # Blend normalized scores
    blended = bf_weight * norm_bf + wf_weight * norm_wf
    
    # Add a small tie-breaker: among similar scores, prefer bins with more remaining
    # to avoid premature closure (small epsilon term)
    scores_valid = blended + 0.01 * (valid_remaining / 100.0)
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
