"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing an item. Higher score = better choice.
    
    Uses a composite heuristic:
    1. Prefer tight fits (small remaining capacity) to avoid fragmentation
    2. But penalize extremely tight fits that might be too wasteful
    3. Consider the relative fit quality based on typical item sizes
    
    For Weibull-distributed items with capacity 100, we use a smooth
    surrogate that balances tightness against creating unusable gaps.
    """
    # Remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return np.full(len(bins), -np.inf)
    
    # For valid bins, compute a score that prefers moderate tightness
    # We want to minimize wasted space but avoid creating tiny gaps
    
    # Method: Use a quadratic penalty on remaining space, but with a
    # floor to avoid over-penalizing very small remainders
    # This creates a smooth preference for "just right" fits
    
    # Scale remaining to [0, 1] relative to capacity
    rem_normalized = remaining / 100.0
    
    # Score: prefer small remaining, but with diminishing returns
    # Using -rem^2 gives strong preference for tight fits
    # But we add a small constant to avoid numerical issues
    
    # Alternative: Use a piecewise linear approach
    # - Very tight (rem < 5%): high score
    # - Moderate (5-30%): moderate score  
    # - Loose (>30%): lower score
    
    # Smooth version using exponential decay:
    # score = -exp(-rem_normalized / tau) where tau controls sensitivity
    # This gives a smooth preference for tighter fits
    
    tau = 0.15  # Time constant - smaller means more aggressive tight-fitting
    scores = -np.exp(-rem_normalized / tau)
    
    # Invalidate bins without enough capacity
    scores[~valid_mask] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
