"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a smooth nonlinear Best Fit heuristic.
    
    Uses a power-law transformation on normalized remaining capacity to create
    a smooth scoring function that prefers tight fits while avoiding the cliff-edge
    behavior of pure Best Fit. This may better handle varying item size distributions.
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
    
    # Normalize remaining capacity by bin capacity (bins are all capacity 100)
    capacity = 100.0
    norm_remaining = remaining[valid_mask] / capacity
    
    # Smooth nonlinear scoring: use a power law that gives higher scores to tighter fits
    # but with diminishing returns, avoiding the sharp preference of pure Best Fit
    # Combine: (1) penalty for remaining space, (2) small bonus for high fill ratio
    # This encourages filling bins toward capacity while still preferring tight fits
    alpha = 2.0
    fill_ratio = 1.0 - norm_remaining  # How full the bin becomes
    # Primary term: penalize remaining space with power law
    penalty = np.power(norm_remaining, alpha)
    # Secondary term: small bonus for filling bins (encourages consolidation)
    fill_bonus = 0.1 * np.power(fill_ratio, 2.0)
    scores[valid_mask] = -penalty + fill_bonus
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
