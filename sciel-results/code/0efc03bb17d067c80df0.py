# EVOLVE-BLOCK-START
import numpy as np

# Global state for online estimation of item distribution
# This allows the heuristic to adapt to the actual Weibull parameters
# used in the evaluation, rather than relying on hardcoded assumptions.
_item_sum = 0.0
_item_count = 0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy:
    Uses the "effective capacity" heuristic:
    Score = -(RemainingSpace - ExpectedNextItemSize)
    
    By estimating the expected next item size online from the observed items,
    this heuristic adapts to the specific Weibull distribution parameters
    (shape and scale) of the test sequence.
    
    The ranking is equivalent to Best Fit (minimizing remaining space),
    but the adaptive estimation ensures the "effective capacity" calculation
    is grounded in the actual data distribution, potentially improving
    tie-breaking or handling edge cases where the hardcoded mean was far off.
    """
    global _item_sum, _item_count
    
    # Update running statistics
    _item_sum += item
    _item_count += 1
    
    # Calculate expected next item size
    # For the first item, we don't have a history, so we use the item itself
    # or a default. Using the current item is a reasonable Markovian assumption.
    if _item_count > 1:
        expected_next = _item_sum / _item_count
    else:
        expected_next = item
        
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins (those with enough remaining capacity for the current item)
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Score: negative of (remaining - expected_next)
    # Maximizing this score is equivalent to minimizing (remaining - expected_next)
    # Since expected_next is constant for all bins in this call,
    # this is equivalent to minimizing 'remaining' (Best Fit).
    scores[valid_mask] = -(remaining - expected_next)
    
    return scores
# EVOLVE-BLOCK-END
