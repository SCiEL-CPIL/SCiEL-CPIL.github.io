"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Modified Best-Fit with Smooth Gap Penalty.
    - Base score favors bins with smaller remaining capacity (Best-Fit).
    - A smooth exponential penalty is applied to very small remaining gaps.
    - This avoids creating "unusable" slivers of space (e.g., < 5 units) 
      that cannot accommodate future items, effectively shifting the preference 
      from "absolute tightest fit" to "best usable fit".
    """
    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores with a very low value for infeasible bins
    scores = np.full(len(bins), -1e18)
    
    if not np.any(feasible):
        return scores
    
    # Calculate the gap (remaining capacity after placing the item) for feasible bins
    gaps = remaining[feasible]
    
    # Parameters for the penalty term
    # score = -gap - K * exp(-gap / sigma)
    # K controls the strength of the penalty for small gaps.
    # sigma controls the width of the penalty region.
    # Tuned to penalize gaps significantly below 5-10 units.
    K = 15.0
    sigma = 4.0
    
    # Calculate the penalty: high for small gaps, approaching 0 for large gaps
    # np.exp handles the vectorization
    penalty = K * np.exp(-gaps / sigma)
    
    # Final score: 
    # -gaps encourages minimizing wasted space (Best-Fit behavior)
    # -penalty discourages creating tiny, likely unusable gaps
    feasible_scores = -gaps - penalty
    
    scores[feasible] = feasible_scores
    
    return scores
# EVOLVE-BLOCK-END
