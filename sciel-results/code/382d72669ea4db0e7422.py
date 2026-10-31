"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Robust Best-Fit.
    1. Identify feasible bins (remaining capacity >= item).
    2. Assign a very low score to infeasible bins to ensure they are never selected.
    3. For feasible bins, use a score that strictly prefers the bin with the 
       smallest remaining capacity after placement (minimizing waste).
    4. Tie-breaking: If multiple bins have the same remaining capacity, 
       the one with the lower index is naturally selected by argmax if scores are identical,
       but we can add a tiny epsilon based on index to break ties deterministically 
       towards the first available bin (stable behavior).
    """
    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores with a very low value for infeasible bins
    # Using -1e18 to ensure they are never picked over any feasible bin
    # (Feasible bins will have scores >= 0 or close to 0 depending on scale)
    scores = np.full(len(bins), -1e18)
    
    if not np.any(feasible):
        return scores

    # For feasible bins, score = -remaining_after_placement
    # This ensures the bin with the smallest 'remaining' gets the highest score.
    # Example: item=5, bins=[10, 20]
    # remaining = [5, 15]
    # scores = [-5, -15]
    # argmax is -5 (index 0), which is Best-Fit.
    
    # To add a tiny deterministic tie-breaker favoring lower indices:
    # If remaining[i] == remaining[j] and i < j, we want score[i] > score[j].
    # We can subtract a tiny epsilon * index.
    # epsilon should be smaller than the smallest possible difference in remaining capacities
    # to not disturb the primary ranking.
    # However, remaining capacities are floats. The difference could be very small.
    # A safer tie-breaker is not to modify the score but rely on the fact that
    # if scores are identical, argmax returns the first occurrence.
    # Numpy's argmax returns the index of the first occurrence of the maximum value.
    # So, if we just use -remaining, ties are broken by lower index automatically.
    
    scores[feasible] = -remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
