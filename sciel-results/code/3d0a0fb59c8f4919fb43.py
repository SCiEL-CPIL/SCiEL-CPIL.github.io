"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Best-fit with a tie-breaking preference for bins that have
    been used less (i.e., have more total capacity remaining relative to
    the bin's original capacity is not available, so we use a combination
    of remaining capacity and a secondary heuristic).
    
    Since we only have the current remaining capacities in `bins`, we use:
    - Primary: minimize remaining capacity after placing the item (best-fit)
    - Secondary: among bins with similar remaining capacity, prefer those
      with more remaining capacity (to avoid fragmenting bins that are
      still relatively full) — this is a subtle point.
    
    Actually, for online bin packing, Best-Fit (BF) is known to perform well.
    The seed program implements Best-Fit by returning -(bins - item), so the
    highest score corresponds to the smallest (bins - item), i.e., the bin
    with the least remaining space after placing the item.
    
    Let's try a more sophisticated approach: use a weighted combination that
    considers both the tightness of the fit and the absolute remaining capacity.
    
    One effective heuristic is to prefer bins where the item fits snugly but
    also to avoid creating many nearly-empty bins. A common improvement is
    to use a "best-fit decreasing" style, but since items arrive online,
    we can't sort them.
    
    Let's try: score = -(bins - item) + epsilon * bins
    where epsilon is small, to break ties by preferring bins with more
    remaining capacity (less fragmented).
    
    Actually, let's think about this differently. The L1 lower bound for
    bin packing is max(ceil(total_size / capacity), max_item_size / capacity).
    The excess is (bins_used - L1_bound) / L1_bound.
    
    To minimize bins used, we want to pack items as tightly as possible.
    Best-fit is a good heuristic. Let's try a variant that also considers
    the ratio of item size to bin capacity.
    
    For now, let's stick with Best-Fit but add a small regularization term
    to prefer bins with slightly more remaining capacity when there are ties.
    """
    # Compute remaining capacity after placing the item
    remaining_after = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining_after >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, use Best-Fit: prefer smallest remaining_after
    # Add a small regularization to prefer bins with more remaining capacity
    # when remaining_after values are very close
    valid_remaining = remaining_after[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Primary score: -remaining_after (best-fit)
    # Secondary score: small bonus for larger bins (less fragmentation)
    # Use a small epsilon to weight the secondary criterion
    epsilon = 1e-6
    scores_valid = -valid_remaining + epsilon * valid_bins
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
