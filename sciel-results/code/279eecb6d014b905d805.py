import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Returns an array of scores. The item goes to the highest-scoring bin.
    
    Strategy: Adaptive blend of Best-Fit and Worst-Fit.
    - For large items, prefer bins with more remaining capacity (Worst-Fit) to preserve space for other large items.
    - For small items, prefer bins with less remaining capacity (Best-Fit) to reduce fragmentation.
    
    Score = w_ratio * (R / C) + (1 - w_ratio) * (1 / R)
    Where R is remaining capacity, C is bin capacity (assumed 100), w_ratio is item/C.
    """
    C = 100.0
    remaining = bins - item
    feasible_mask = remaining >= 0
    
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    if not np.any(feasible_mask):
        return scores

    R = remaining[feasible_mask]
    
    # Avoid division by zero
    R_safe = np.where(R > 1e-9, R, 1e-9)
    
    w_ratio = item / C
    
    # Weighted combination
    # Worst-Fit component: R (higher is better)
    # Best-Fit component: 1/R (higher is better for smaller R)
    
    # Normalize R to [0, 1] roughly? R is in [0, C].
    # 1/R is in [1/C, inf].
    # To make them comparable, we might need scaling.
    # Let's scale 1/R by C so it's in [1, C].
    # Worst-Fit term: R
    # Best-Fit term: C / R
    
    wf_score = R
    bf_score = C / R_safe
    
    # Blend
    # If item is large (w_ratio close to 1), use wf_score
    # If item is small (w_ratio close to 0), use bf_score
    
    score = w_ratio * wf_score + (1 - w_ratio) * bf_score
    
    scores[feasible_mask] = score
    
    return scores
