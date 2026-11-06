"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a rank-based best-fit heuristic.
    
    The evaluator picks the bin with the highest score among bins with enough remaining capacity.
    We want to minimize the number of bins used (excess over L1 lower bound).
    
    Strategy:
    - Use rank-based scoring to favor bins with smaller remaining spaces
    - This reduces fragmentation by preferring tight fits
    - Rank-based scoring is robust to distribution shape
    """
    deadline = time.time() + 0.5  # 0.5 second budget with safety margin
    
    remaining = bins - item
    
    # Filter: only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # If no valid bins, return scores that will result in no selection
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    # Rank-based scoring: score is negative count of bins with smaller remaining space
    # This means bins with the smallest remaining space get the highest score (closest to 0)
    # Bins with larger remaining space get more negative scores
    # This is equivalent to: score = -(number of bins with remaining < current bin's remaining)
    
    # Use argsort to get ranks
    sorted_indices = np.argsort(valid_remaining)
    ranks = np.empty(n_valid, dtype=np.float64)
    ranks[sorted_indices] = np.arange(n_valid)
    
    # Score: prefer bins with smaller remaining space
    # Higher score for smaller remaining (rank 0 is smallest)
    rank_score = -ranks
    
    # Add a small penalty based on the actual remaining space to break ties
    # and provide gradient information
    min_remaining = np.min(valid_remaining)
    max_remaining = np.max(valid_remaining)
    if max_remaining > min_remaining:
        normalized_remaining = (valid_remaining - min_remaining) / (max_remaining - min_remaining)
    else:
        normalized_remaining = np.zeros(n_valid)
    
    # Small penalty for larger remaining spaces
    penalty = 0.01 * normalized_remaining
    
    final_scores = rank_score - penalty
    
    scores[valid_mask] = final_scores
    
    # Check deadline
    if time.time() > deadline:
        # Return whatever we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
