"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Combine multiple heuristics to improve bin packing.
    - Best-fit: prefer bins that leave the least remaining capacity
    - First-fit: prefer bins that are already somewhat filled
    - Balance: avoid creating too many partially-filled bins
    
    We use a weighted combination that favors:
    1. Minimizing waste (best-fit)
    2. Keeping bins well-utilized when possible
    3. A slight preference for bins with moderate utilization to balance load
    """
    remaining = bins - item
    valid = remaining >= 0
    
    # Base score: negative remaining capacity (best-fit)
    score = -remaining.copy()
    
    # Only score valid bins
    score[~valid] = -np.inf
    
    # For valid bins, add a bonus for bins that are already somewhat filled
    # This encourages using bins that are already in use rather than opening new ones
    # when there's a choice
    utilization = item / np.where(bins > 0, bins, 1.0)
    valid_utilization = np.where(valid, utilization, 0.0)
    
    # Small bonus for higher utilization (prefer filling existing bins)
    score[valid] += 0.1 * valid_utilization[valid]
    
    # Penalize bins that would become almost full but not full (wasteful)
    # If remaining after placing is very small but not zero, it's fine
    # But if remaining is moderate, we might prefer other options
    post_remaining = remaining[valid]
    # Add a small penalty for large remaining capacity to prefer tighter fits
    score[valid] -= 0.05 * post_remaining
    
    return score
# EVOLVE-BLOCK-END
