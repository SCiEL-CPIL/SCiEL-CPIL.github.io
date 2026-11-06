"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Smooth best-fit surrogate with adaptive fragment penalty.
    
    We use a smooth scoring function that:
    1. Primarily minimizes remaining capacity (best-fit behavior)
    2. Uses a sublinear penalty (power < 1) to reduce sensitivity to large remainders
    3. Applies a smooth penalty for very small remainders (fragments)
       using a smooth step function instead of a hard threshold
    
    The score is: score = -(rem + eps)^p - smooth_fragment_penalty(rem)
    
    Where:
    - p = 0.6 (slightly more aggressive than sqrt, less than linear)
    - smooth_fragment_penalty uses a smoothstep-like function around rem < threshold
    - This avoids the kink in the hard-threshold fragment penalty
    
    For Weibull distributions with capacity 100, this balances:
    - Filling bins efficiently (minimize bins used)
    - Avoiding tiny unusable fragments
    - Smoothness for better generalization
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        rem = remaining[valid]
        eps = 1e-9
        
        # Harmonic mean approach: penalizes both tiny gaps and large remaining spaces
        # Harmonic mean of rem and item: 2 * rem * item / (rem + item)
        # This is maximized when rem ≈ item (balanced fit)
        # We want to MINIMIZE waste, so we use negative harmonic mean
        
        # Compute harmonic mean of remaining capacity and item size
        # For very small rem, harmonic mean → 0 (bad)
        # For very large rem, harmonic mean → item (wastes space)
        # Optimal when rem ≈ item
        
        harmonic_mean = 2.0 * rem * item / (rem + item + eps)
        
        # Score: negative harmonic mean (higher is better = smaller waste)
        # Add a small best-fit term to break ties
        # p=0.5 gives sqrt behavior, less aggressive than linear
        p = 0.5
        best_fit_term = -(rem + eps) ** p * 0.1  # Small weight to prefer tighter fits
        
        scores[valid] = -harmonic_mean + best_fit_term
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Fallback: simple best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
