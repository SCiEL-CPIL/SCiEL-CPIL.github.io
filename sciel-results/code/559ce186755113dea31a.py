import numpy as np

C = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    T = 50.0
    width = 10.0
    
    feasible = bins >= item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(feasible):
        return scores
    
    # Calculate weight for Worst Fit vs Best Fit
    # w close to 1 -> Worst Fit, w close to 0 -> Best Fit
    # Sigmoid centered at T
    t = (item - T) / width
    w = 1.0 / (1.0 + np.exp(-t))
    
    # Normalized scores for feasible bins
    # Best Fit score: maximize (C - bins) which is equivalent to minimizing bins
    # Worst Fit score: maximize (bins - item) which is equivalent to maximizing bins
    # Both are scaled to be positive and comparable in magnitude [0, C-item]
    
    b = bins[feasible]
    score_bf = C - b
    score_wf = b - item
    
    # Blend
    blended = (1.0 - w) * score_bf + w * score_wf
    
    # Penalty for fragmentation: penalize bins with remaining capacity close to C/2
    # This encourages bins to be either very full (low remaining) or very empty (high remaining)
    # Avoids the "middle" state which is hard to utilize for future items.
    center = C / 2.0
    sigma = 15.0
    penalty = -20.0 * np.exp(-((b - center) ** 2) / (2 * sigma ** 2))
    
    scores[feasible] = blended + penalty
    
    return scores
