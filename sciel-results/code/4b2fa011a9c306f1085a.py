import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins contains remaining capacities
    # item is the size of the current item
    # Capacity is implicitly 100 based on the task description, 
    # but we should probably infer it or hardcode it if not passed.
    # The task says "capacity 100".
    
    capacity = 100.0
    
    # Identify valid bins
    valid_mask = bins >= item
    
    # Initialize scores with a very low value for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    # Strategy: Hybrid BF/WF
    # Threshold: 0.5 * capacity
    threshold = 0.5 * capacity
    
    if item > threshold:
        # Large item: Worst Fit (maximize remaining space)
        # We want to select the bin with the MAXIMUM remaining space among valid bins.
        # So score should be proportional to bins.
        scores[valid_mask] = bins[valid_mask]
    else:
        # Small/Medium item: Best Fit (minimize remaining space)
        # We want to select the bin with the MINIMUM remaining space among valid bins.
        # So score should be inversely proportional to bins.
        # Using -bins works because higher score is better.
        scores[valid_mask] = -bins[valid_mask]
        
    return scores
