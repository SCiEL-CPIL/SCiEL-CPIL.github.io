import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the given item.
    The item goes to the highest-scoring bin.
    Lower total bins used is better (lower excess over L1 lower bound).
    """
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Strategy: Probabilistic risk assessment based on global variance of remaining capacity.
    # We score bins such that we prefer bins that align with the global statistical trend.
    # Specifically, we use the negative variance of the remaining capacities of valid bins.
    # Wait, the variance is a scalar for the whole set. The directive says:
    # "scoring bins based on the negative variance of the remaining capacity distribution across all currently open bins"
    # This implies a global score? Or is it a per-bin score derived from the distribution?
    # "prioritizes bins that align with the global statistical trend of the load"
    
    # Let's interpret this as:
    # Calculate the mean and variance of remaining capacities of valid bins.
    # A bin that is "close" to the mean remaining capacity might be "safer" or "more representative".
    # Or, perhaps we want to minimize the variance of the *final* state?
    
    # Let's try a score that rewards bins whose remaining capacity is close to the median/mean of valid bins.
    # This would tend to keep bins balanced, potentially reducing fragmentation.
    
    valid_remaining = remaining[valid_mask]
    if len(valid_remaining) == 0:
        return scores
        
    # If only one valid bin, it's the only choice
    if len(valid_remaining) == 1:
        scores[valid_mask] = 0.0
        return scores

    mean_rem = np.mean(valid_remaining)
    # Use distance from mean as a penalty?
    # score = -|remaining - mean|
    # This would pick the bin closest to the average fill level.
    
    # Alternatively, the directive mentions "negative variance".
    # Maybe the score for a bin is related to how much it contributes to the variance?
    # Variance = E[(X - mu)^2].
    # If we pick a bin, we reduce its remaining capacity by `item`.
    # This changes the distribution.
    
    # Let's stick to a simpler interpretation first:
    # Score = - (remaining - mean)^2
    # This selects the bin with remaining capacity closest to the mean.
    
    scores[valid_mask] = -(valid_remaining - mean_rem)**2
    
    # But wait, the previous successful strategy was a blend of Best Fit and Worst Fit.
    # The directive says "KEEP this design and investigate improvements within its family."
    # The current design is a smooth blend of Best Fit (w=0) and Worst Fit (w=1).
    # The strategy to try is "Model the bin selection as a probabilistic risk assessment...".
    # This suggests adding a variance-based component or replacing the heuristic with it.
    
    # Let's blend the existing heuristic with the variance-based score.
    # Existing heuristic:
    low = 50.0
    high = 90.0
    
    if item <= low:
        w = 0.0
    elif item >= high:
        w = 1.0
    else:
        t = (item - low) / (high - low)
        w = t * t * (3.0 - 2.0 * t)
        
    heuristic_scores = (2.0 * w - 1.0) * remaining[valid_mask]
    
    # Normalize heuristic scores to [0, 1] or similar to blend?
    # Or just add them?
    # The scales are different.
    # heuristic_scores range: roughly [-100, 100]
    # variance_scores range: roughly [-10000, 0]
    
    # Let's normalize both to have similar impact.
    # Min-max normalize heuristic scores
    h_min = np.min(heuristic_scores)
    h_max = np.max(heuristic_scores)
    if h_max > h_min:
        h_norm = (heuristic_scores - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(heuristic_scores)
        
    v_scores = -(valid_remaining - mean_rem)**2
    v_min = np.min(v_scores)
    v_max = np.max(v_scores)
    if v_max > v_min:
        v_norm = (v_scores - v_min) / (v_max - v_min)
    else:
        v_norm = np.zeros_like(v_scores)
        
    # Blend factor: how much to trust the variance heuristic?
    # Maybe 0.5?
    blend = 0.5
    final_scores = (1 - blend) * h_norm + blend * v_norm
    
    scores[valid_mask] = final_scores
    
    return scores
