"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a combination of:
    1. Best-fit preference (minimize leftover space)
    2. Penalize bins that are too empty (waste)
    3. Prefer bins where the item fills a significant portion
    
    The score balances tight fit with avoiding creating many nearly-empty bins.
    """
    rem = bins - item
    
    # Mask for valid bins (enough capacity)
    valid = rem >= 0
    
    # For invalid bins, assign very low score
    scores = np.full_like(rem, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    valid_rem = rem[valid]
    valid_bins = bins[valid]
    
    # Component 1: Best-fit (negative remainder, so tighter fit = higher score)
    # Normalize by item size to make it scale-invariant
    fit_score = -valid_rem / max(item, 1e-10)
    
    # Component 2: Fill ratio - prefer bins where item takes significant portion
    # fill_ratio = item / bin_capacity, higher is better (up to 1.0)
    fill_ratio = item / np.maximum(valid_bins, 1e-10)
    
    # Component 3: Penalty for creating very small remainders that can't be used
    # If remainder is very small relative to typical item sizes, it's wasted
    # Use a soft penalty: penalty increases as remainder approaches 0
    # But we want to AVOID large remainders too, so this is tricky
    # Instead, use a quadratic term that penalizes both very small and very large remainders
    # Optimal remainder is around some fraction of item size
    
    # Combined score: weight fit and fill ratio
    # Higher fill ratio means the bin was already partially filled, which is good
    # Tighter fit (smaller rem) is also good
    
    # Use a weighted combination
    w_fit = 1.0
    w_fill = 0.3
    
    combined = w_fit * fit_score + w_fill * fill_ratio
    
    # Additional refinement: if remainder is very small (< 5% of item), 
    # it's essentially wasted, so add a small bonus for "perfect" fits
    # but not too much to override other considerations
    perfect_fit_bonus = np.where(valid_rem < 0.05 * item, 0.1, 0.0)
    
    scores[valid] = combined + perfect_fit_bonus
    
    return scores
# EVOLVE-BLOCK-END
