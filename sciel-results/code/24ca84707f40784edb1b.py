"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a refined Best Fit heuristic with smooth nonlinear scoring.
    
    Primary objective: minimize remaining capacity after placement (Best Fit).
    Secondary objective: smooth penalty that discourages leaving very small slivers
    (which may be unusable for future items) while still strongly preferring tighter fits.
    
    The score combines:
      1. -remaining (dominant term, ensures tightest fit wins)
      2. A small smooth bonus for bins where remaining is not too small relative to capacity,
         which helps avoid pathological fragmentation on Weibull-distributed items.
    """
    start_time = time.time()
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Best Fit core: smaller remaining -> higher score
    # Use a smooth transformation that preserves ordering but adds mild nonlinearity
    rem_valid = remaining[valid_mask]
    
    # Normalize remaining by max bin capacity for scale invariance
    max_cap = np.max(bins) if len(bins) > 0 else 1.0
    if max_cap <= 0:
        max_cap = 1.0
    
    rem_norm = rem_valid / max_cap
    
    # Primary: negative remaining (tighter fit = higher score)
    # Use a power law with exponent slightly > 1 to amplify the preference for tight fits
    # while keeping the function smooth and differentiable
    alpha = 1.2
    primary_score = -np.power(rem_norm, alpha)
    
    # Secondary: mild smooth bonus that reduces the penalty for moderately small remainders
    # This helps when multiple bins have similar remaining capacity
    # Use a smooth approximation that peaks around rem_norm ~ 0.1-0.2
    # and gently discourages both very small slivers and very large remainders
    # The bonus is small enough not to override the primary Best Fit signal
    beta = 5.0
    gamma = 0.05
    # Smooth bump: exp(-beta * (rem_norm - 0.15)^2) centered at 15% remaining
    secondary_score = gamma * np.exp(-beta * (rem_norm - 0.15) ** 2)
    
    # Combine: primary dominates, secondary provides smooth tie-breaking
    combined = primary_score + secondary_score
    
    scores[valid_mask] = combined
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
