import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
        remaining_after = bins - item
        scores = np.full(len(bins), -np.inf, dtype=float)
        
        valid_mask = remaining_after >= 0
        if not np.any(valid_mask):
            return scores
            
        r_new = remaining_after[valid_mask]
        b_valid = bins[valid_mask]
        
        # Base score: Best-Fit
        # Higher score for smaller remaining capacity
        base_scores = -r_new
        
        # Fragmentation penalty
        # Penalize bins where the remaining space is in a "dead zone"
        # Dead zone: not tight enough to be negligible, not large enough to be useful
        # Assuming capacity ~100
        # Let's define dead zone as 10% to 40% of the bin's original capacity
        # Or just fixed values since capacity is likely constant per bin type?
        # The prompt says "capacity 100". So bins are likely all 100.
        
        capacity = 100.0
        low_bound = 0.1 * capacity  # 10
        high_bound = 0.4 * capacity # 40
        
        penalty = np.zeros_like(r_new)
        
        in_zone = (r_new > low_bound) & (r_new < high_bound)
        
        if np.any(in_zone):
            # Center of the dead zone
            center = (low_bound + high_bound) / 2.0
            # Max penalty
            max_penalty = 20.0 # Make it significant
            
            # Distance from center
            dist = np.abs(r_new[in_zone] - center)
            half_width = (high_bound - low_bound) / 2.0
            
            # Normalized distance (0 at center, 1 at edges)
            norm_dist = dist / half_width
            
            # Penalty is max at center, 0 at edges
            # Using a cosine or quadratic taper for smoothness
            # p = max_penalty * (1 - norm_dist**2)
            p_val = max_penalty * (1 - norm_dist**2)
            
            penalty[in_zone] = p_val
        
        final_scores = base_scores - penalty
        scores[valid_mask] = final_scores
        
        return scores
