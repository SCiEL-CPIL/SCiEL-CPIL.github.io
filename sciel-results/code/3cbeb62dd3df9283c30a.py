import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback: Best Fit (minimize remaining)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Calculate gap for valid bins
    g = bins - item
    
    # Global state: mean remaining capacity (0 to 100)
    # This proxies for how "full" the system is.
    # High mean_rem -> Loose/Early -> Favor Worst Fit (large g)
    # Low mean_rem -> Tight/Late -> Favor Best Fit (small g)
    mean_rem = np.mean(bins)
    w = mean_rem / 100.0
    
    # Adaptive weight: -1 (Best Fit) to 1 (Worst Fit)
    weight = 2.0 * w - 1.0
    
    # Base score: Adaptive Fit
    # If w > 0.5, weight > 0, score increases with g (Worst Fit)
    # If w < 0.5, weight < 0, score decreases with g (Best Fit)
    base_score = weight * g
    
    # Polarization Score:
    # We prefer bins that leave a remainder close to 0 (full) or 100 (empty).
    # This avoids creating "medium" gaps which are often dead space.
    # polarization is 1 if g is 0 or 100, 0 if g is 50.
    # Formula: 1 - |2*(g/100) - 1|
    # Normalize g to [0, 1]
    g_norm = g / 100.0
    polarization = 1.0 - np.abs(2.0 * g_norm - 1.0)
    
    # Combine scores
    # The polarization term is a small nudge to break ties or adjust mid-game behavior.
    # We scale it down so it doesn't override the primary fit strategy.
    # In mid-game (w~0.5), base_score is ~0, so polarization dominates.
    # In early/late game, base_score dominates.
    final_scores = base_score + 0.1 * polarization
    
    # Apply validity mask
    scores = np.where(valid, final_scores, -np.inf)
    
    return scores
