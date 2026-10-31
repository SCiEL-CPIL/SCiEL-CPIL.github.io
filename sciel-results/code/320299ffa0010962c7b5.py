"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Polarization-aware Best Fit
    - Prefer bins where the item fits tightly (best-fit) to reduce fragmentation
    - Add a polarization term: strongly prefer bins that will become very full or very empty
    - This reduces the number of 'half-full' bins that are hard to pack efficiently
    
    The score is designed so that higher = better for the caller.
    """
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=np.float64)
    
    # Remaining capacity after placing item
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # For invalid bins, set score to -inf so they are never chosen
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Get remaining capacity for valid bins
    rem_valid = remaining[valid_mask]
    
    # Best-fit component: prefer smaller remaining (tighter fit)
    # Normalize to [0, 1] range where 1 = tightest fit
    min_rem = np.min(rem_valid)
    max_rem = np.max(rem_valid)
    if max_rem - min_rem > 1e-12:
        # Inverse normalized remaining: 1 when tight, 0 when loose
        best_fit_score = 1.0 - (rem_valid - min_rem) / (max_rem - min_rem)
    else:
        best_fit_score = np.ones_like(rem_valid)
    
    # Polarization term: reward bins that will be very full (small remaining) or very empty (large remaining)
    # Penalty for middle-range remaining capacities
    # Use a quadratic penalty centered at the midpoint of the valid range
    mid = (min_rem + max_rem) / 2.0
    half_range = (max_rem - min_rem) / 2.0
    if half_range > 1e-12:
        # Distance from midpoint, normalized to [0, 1]
        dist_from_mid = np.abs(rem_valid - mid) / half_range
        # Polarization score: higher when far from midpoint
        polarization_score = dist_from_mid
    else:
        polarization_score = np.ones_like(rem_valid)
    
    # Combine: weighted sum of best-fit and polarization
    # Weight best-fit more heavily since it's the primary heuristic
    alpha = 0.7
    beta = 0.3
    combined = alpha * best_fit_score + beta * polarization_score
    
    # Assign scores only to valid bins
    scores[valid_mask] = combined
    
    return scores
# EVOLVE-BLOCK-END
