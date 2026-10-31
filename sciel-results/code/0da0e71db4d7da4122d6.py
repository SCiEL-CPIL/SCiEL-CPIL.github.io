import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Modified Best Fit with fragmentation penalty:
    # Base score: -remaining (prefer tight fits)
    # Penalty: avoid creating very small remainders that become unusable
    # Weibull items with capacity 100: penalize remainders < 15
    base_scores = -remaining[valid_mask]
    frag_penalty = 10.0 * np.maximum(0.0, 15.0 - remaining[valid_mask])
    scores[valid_mask] = base_scores - frag_penalty
    
    return scores
