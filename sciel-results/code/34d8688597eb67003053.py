"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for tracking item statistics
_item_history = []
_MEDIAN_WINDOW = 50  # Keep last 50 items for median estimation


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Combine best-fit with a "Next-Item Fit" proxy.
    
    Primary score: -waste (best-fit, minimize remaining space after placement)
    Secondary score: -squared difference between remaining_after and estimated
    median item size. This encourages leaving space that can accommodate typical
    future items, reducing fragmentation.
    
    The combined score balances tight packing (best-fit) with future compatibility.
    """
    global _item_history
    
    # Update item history
    _item_history.append(item)
    if len(_item_history) > _MEDIAN_WINDOW:
        _item_history = _item_history[-_MEDIAN_WINDOW:]
    
    # Estimate median item size from history
    if len(_item_history) >= 1:
        median_item = float(np.median(_item_history))
    else:
        median_item = item
    
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    if np.any(valid):
        waste = remaining[valid]
        
        # Primary: best-fit (minimize waste)
        # Use -waste so higher score = less waste
        primary_score = -waste
        
        # Secondary: Next-Item Fit proxy
        # Score based on how well the remaining space matches typical future items
        # We want remaining_after to be close to median_item (can fit one more typical item)
        # But not too small (fragmentation) and not too large (wasted capacity)
        # Use negative squared difference: -((waste - median_item) / median_item)^2
        # This gives a smooth, bounded penalty
        if median_item > 0:
            normalized_diff = (waste - median_item) / median_item
            secondary_score = -normalized_diff ** 2
        else:
            secondary_score = np.zeros_like(waste)
        
        # Combine: primary dominates, secondary breaks ties and guides near-ties
        # Weight: primary gets most weight, secondary is a gentle nudge
        scores[valid] = primary_score + 0.1 * secondary_score
        
        # Tiny tiebreaker for numerical stability
        scores[valid] += 1e-8 * bins[valid]
    
    return scores


# EVOLVE-BLOCK-END
