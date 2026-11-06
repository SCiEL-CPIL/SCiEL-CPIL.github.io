import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    if np.any(valid):
        rem_valid = remaining[valid]
        
        # Score based on how well the item fits in the bin
        # We want to reward fits where the item uses a significant portion of remaining space
        # but not so tight that it leaves unusable fragments
        
        # Fraction of remaining space used by the item
        # If remaining is 0, this would be problematic, but valid ensures remaining >= 0
        # and if remaining == 0, the item must be 0 too (or we skip)
        with np.errstate(divide='ignore', invalid='ignore'):
            fill_ratio = item / (rem_valid + item)  # fraction of new total that was item
        
        # fill_ratio is between 0 and 1
        # We want to maximize fill_ratio (tighter fit is better)
        # But penalize extremely tight fits where remaining is very small
        
        # Primary score: reward higher fill ratio
        base_score = fill_ratio
        
        # Penalty for very small remaining space (fragmentation)
        # If remaining < threshold, apply a penalty
        threshold = 10.0
        small_rem = rem_valid < threshold
        if np.any(small_rem):
            # Penalty scales with how small the remaining space is
            penalty = 0.3 * (threshold - rem_valid[small_rem]) / threshold
            idx_small = np.where(valid)[0][small_rem]
            base_score[small_rem] -= penalty
        
        # Bonus for moderate remaining space (flexibility)
        # If remaining is between 10 and 30, give a small bonus
        moderate = (rem_valid >= threshold) & (rem_valid <= 30)
        if np.any(moderate):
            bonus = 0.05
            idx_mod = np.where(valid)[0][moderate]
            base_score[moderate] += bonus
        
        # For bins with very large remaining space, slight penalty (wasted capacity)
        large = rem_valid > 50
        if np.any(large):
            penalty_large = 0.02
            idx_large = np.where(valid)[0][large]
            base_score[large] -= penalty_large
        
        scores[valid] = base_score
    return scores
