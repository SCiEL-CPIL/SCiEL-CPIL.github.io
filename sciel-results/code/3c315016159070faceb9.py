"""Improved program: Best-Fit with adaptive softening for small items."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing the given item.
    bins: array of remaining capacities.
    Returns: array of scores, higher is better.
    Only bins with enough capacity are scored; others get -inf.
    
    Strategy:
    - For large items, use Best Fit (minimize remaining capacity).
    - For small items, soften the preference to avoid over-consolidating small fragments
      which might leave suboptimal configurations for future items.
      Specifically, we blend Best Fit with a neutral score based on a sigmoid function
      of the item size.
    """
    remaining = bins
    mask = remaining >= item
    
    if not np.any(mask):
        return -np.inf * np.ones_like(remaining)
    
    # We want to minimize (remaining - item) for large items (Best Fit).
    # For small items, we want to be less aggressive, perhaps slightly preferring
    # bins that are not the absolute tightest, to preserve tight bins?
    # Actually, the previous finding was:
    # "Inverting the hybrid heuristic to use Worst Fit for large items ... and Best Fit for small items"
    # Wait, the finding says:
    # "Inverting the hybrid heuristic to use Worst Fit for large items (to keep bins open for other large items) and Best Fit for small items (to consolidate) addresses the likely cause of the parent's poor performance, which used the opposite logic."
    # Measured: 0.05902 -> 0.9998.
    # This suggests the parent (which was the current program) used:
    # "Best Fit for large items" and "Worst Fit (or neutral) for small items".
    # And the improvement came from switching to:
    # "Worst Fit for large items" and "Best Fit for small items".
    
    # Let's re-read carefully.
    # Current program logic:
    # sig = sigmoid(k * (item - threshold))
    # coeff = -sig
    # scores = coeff * remaining
    # If item is large, sig ~ 1, coeff ~ -1. scores ~ -remaining.
    # Maximizing -remaining is equivalent to minimizing remaining. This is Best Fit.
    # If item is small, sig ~ 0, coeff ~ 0. scores ~ 0. This is Neutral (any bin).
    
    # The finding says the parent used "Best Fit for large" and "opposite logic" for small.
    # The parent's small logic was "Neutral/Worst Fit-like" (coeff 0).
    # The improvement used "Worst Fit for large" and "Best Fit for small".
    
    # So we should implement:
    # Large items -> Worst Fit (Maximize remaining capacity).
    # Small items -> Best Fit (Minimize remaining capacity).
    
    # Let's verify the logic.
    # Worst Fit for large items: If I have a large item, putting it in the largest available bin
    # leaves the smaller bins intact. This might be good if future items are small and can fit in the small bins.
    # Best Fit for small items: If I have a small item, putting it in the tightest available bin
    # "closes" that bin or fills it up, potentially freeing up the larger bins for future large items.
    
    # This is a known heuristic: "Worst Fit Decreasing" is often better for large items if the sequence is decreasing?
    # But here the sequence is random Weibull.
    
    # Let's implement this inversion.
    # We want:
    # Large item -> Score proportional to +remaining (Worst Fit).
    # Small item -> Score proportional to -remaining (Best Fit).
    
    # Transition:
    # Use a sigmoid to blend between -remaining (Best Fit) and +remaining (Worst Fit).
    # Let w be the weight for Worst Fit.
    # w = sigmoid(k * (item - threshold))
    # If item is large, w ~ 1. Score ~ remaining. (Worst Fit)
    # If item is small, w ~ 0. Score ~ -remaining. (Best Fit)
    
    # Score = w * remaining + (1 - w) * (-remaining)
    # Score = (2w - 1) * remaining
    
    # If w=0 (small), Score = -remaining. Best Fit.
    # If w=1 (large), Score = +remaining. Worst Fit.
    
    # Parameters:
    # threshold: item size where we switch. Capacity is 100.
    # Maybe threshold = 10 or 20?
    # Let's try threshold = 10.
    # k: steepness.
    
    threshold = 10.0
    k = 0.5
    
    sig = 1.0 / (1.0 + np.exp(-k * (item - threshold)))
    # sig is 0 for small, 1 for large.
    
    coeff = 2.0 * sig - 1.0
    # coeff is -1 for small, +1 for large.
    
    scores = coeff * remaining
    
    # Mask out bins that don't fit.
    scores[~mask] = -np.inf
    
    return scores

# EVOLVE-BLOCK-END
