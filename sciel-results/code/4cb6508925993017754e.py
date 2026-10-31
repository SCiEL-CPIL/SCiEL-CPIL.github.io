"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy: Waste-Pressure Heuristic
    - Models the 'cost' of remaining space using a non-linear pressure function.
    - Pressure P(r) = r^2 / (r + epsilon) approximates r for small r and r-eps for large r.
    - This creates a dynamic tension:
        * Small remainders (tight fit): Low pressure, high score.
        * Large remainders (loose fit): High pressure, low score.
    - This encourages consolidating small items into nearly full bins (minimizing fragmentation)
      while discouraging the use of bins with large empty spaces unless the item fits very tightly,
      thereby preserving large contiguous spaces for future large items.
    """
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # The pressure function: P(r) = r^2 / (r + epsilon)
    # epsilon controls the transition point.
    # If epsilon is small (e.g., 1.0), the function is close to r for r >> 1.
    # If epsilon is large, it saturates faster.
    # We want to penalize large remainders heavily.
    # Let's use a dynamic epsilon based on the item size to scale the sensitivity.
    # Smaller items should be more sensitive to large remainders? 
    # Actually, the absolute remaining space is what matters for future large items.
    # A fixed epsilon around 10-20 (10-20% of capacity 100) is a reasonable heuristic.
    # Let's use epsilon = 15.0.
    
    epsilon = 15.0
    
    # Calculate pressure for each valid bin
    # P(r) = r^2 / (r + epsilon)
    # This is 0 at r=0, and grows quadratically for small r, linearly for large r.
    # We want to MINIMIZE pressure, so score = -Pressure.
    
    pressure = (valid_remaining * valid_remaining) / (valid_remaining + epsilon)
    
    # To make the scores more distinct and avoid floating point issues, 
    # we can also add a small linear term to ensure strict ordering for very small remainders
    # where quadratic might be too flat.
    # score = -pressure - alpha * valid_remaining
    # However, the quadratic term already dominates for r > epsilon.
    # For r < epsilon, P(r) ~ r^2/epsilon, which is small.
    
    # Let's refine: We want to strongly prefer the tightest fit among "good" fits.
    # The pressure function does this.
    
    scores[valid_indices] = -pressure
    
    return scores
# EVOLVE-BLOCK-END
