import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_after = bins - item
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    valid_mask = remaining_after >= 0
    if not np.any(valid_mask):
        return scores
        
    r_new = remaining_after[valid_mask]
    b_valid = bins[valid_mask]
    
    # Base score: Best-Fit (prefer smaller remaining capacity)
    base_scores = -r_new
    
    # Fragmentation-aware penalty: penalize bins where item/remaining_capacity_before
    # is in a moderate range, creating a medium fragment with low utility.
    # We use a smooth Gaussian-like penalty centered at the worst-case ratio.
    
    # Ratio of item to bin capacity before placement
    ratios = item / b_valid
    
    # The worst ratio is around 0.35 (center of 0.2-0.5 zone)
    # Use a smooth penalty that is zero at ratio=0 and ratio=1,
    # peaking at the worst ratio.
    
    # Model: penalty = A * exp(-((ratio - center) / sigma)^2)
    # but only for ratios in a reasonable range
    center = 0.35
    sigma = 0.12
    
    # Compute penalty for all valid bins
    diff = ratios - center
    penalty = 12.0 * np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Suppress penalty for very small ratios (item << bin) since those are fine
    # and for very large ratios (item ~ bin) since those are tight fits
    small_ratio_mask = ratios < 0.15
    penalty[small_ratio_mask] *= 0.3
    
    # Also slightly reduce penalty for very tight fits (ratio > 0.7)
    tight_mask = ratios > 0.7
    penalty[tight_mask] *= 0.5
    
    final_scores = base_scores - penalty
    scores[valid_mask] = final_scores
    
    return scores
