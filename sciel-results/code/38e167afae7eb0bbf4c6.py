import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_after = bins - item
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    valid_mask = remaining_after >= 0
    if not np.any(valid_mask):
        return scores
        
    r_new = remaining_after[valid_mask]
    
    # Base score: Best-Fit (prefer smaller remaining capacity)
    base_scores = -r_new
    
    # Fragmentation penalty
    # We penalize remaining capacities in the range [10, 30]
    # This range represents "small but not negligible" fragments
    # that are likely unusable for future items but waste space.
    penalty = np.zeros_like(r_new)
    
    # Define the penalty zone
    lower_bound = 10.0
    upper_bound = 30.0
    center = 20.0
    max_penalty = 25.0
    
    in_zone = (r_new >= lower_bound) & (r_new <= upper_bound)
    
    if np.any(in_zone):
        # Calculate distance from center
        dist = np.abs(r_new[in_zone] - center)
        # Normalized distance 0 at center, 1 at bounds
        norm_dist = dist / (upper_bound - center)
        # Penalty is max at center, 0 at bounds
        p_val = max_penalty * (1 - norm_dist)
        penalty[in_zone] = p_val
        
    final_scores = base_scores - penalty
    scores[valid_mask] = final_scores
    
    return scores
