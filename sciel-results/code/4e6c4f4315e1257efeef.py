import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    
    Uses a fluid-flow-inspired heuristic: prioritize bins where the item
    constitutes a large fraction of the remaining capacity, minimizing
    relative waste. Combined with a best-fit bias for consolidation.
    """
    n_bins = len(bins)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    valid_remaining = bins[valid_mask]
    
    # Fluid flow score: item / remaining_capacity
    # Higher means the item fills a larger fraction of the bin
    # This minimizes the "relative waste" of opening/using a bin
    fluid_score = item / valid_remaining
    
    # Best fit component: minimize remaining after placement
    # remaining_after = valid_remaining - item
    # Score: -remaining_after (lower remaining is better)
    remaining_after = valid_remaining - item
    best_fit_score = -remaining_after
    
    # Combine: primarily use fluid score, with a secondary best-fit bias
    # The fluid score naturally handles the small/large item transition:
    # - For small items: fluid_score varies a lot, favoring bins where item is large fraction
    # - For large items: fluid_score is high for all valid bins, so best-fit breaks ties
    
    # Normalize the scores to be comparable
    # fluid_score ranges from item/max_remaining to item/min_remaining
    # best_fit_score ranges from -max_remaining_after to -min_remaining_after
    
    # Use a weighted combination
    # Weight the fluid score more heavily for small items
    # For large items, best-fit becomes more important
    
    # Smooth transition based on item size relative to capacity (100)
    # When item is small (< 20), fluid score dominates
    # When item is large (> 60), best-fit dominates
    
    item_ratio = item / 100.0
    
    # Blend weight: 1.0 means fluid dominates, 0.0 means best-fit dominates
    # Smooth logistic transition centered at 0.4 with width 0.15
    blend = 1.0 / (1.0 + np.exp((item_ratio - 0.4) / 0.15))
    
    # Normalize each component to have similar scale
    # fluid_score: typical range [0.01, 1.0] for items 1-100
    # best_fit_score: typical range [-99, 0]
    
    # Scale best_fit to be comparable to fluid_score
    # Divide by 100 to bring to similar range
    best_fit_scaled = best_fit_score / 100.0
    
    # Combine
    combined = blend * fluid_score + (1.0 - blend) * best_fit_scaled
    
    scores[valid_mask] = combined
    
    return scores
