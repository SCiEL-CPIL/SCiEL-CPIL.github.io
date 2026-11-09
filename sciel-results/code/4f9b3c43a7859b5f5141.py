import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacities
    remaining = bins - item
    
    # Only consider bins that can fit the item
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores

    # Estimate next item size. 
    # Using current item as a proxy for the scale of the distribution.
    # Weibull is right-skewed, so next item might be smaller, 
    # but current item is a robust sample.
    next_item_est = item
    
    # Value of keeping a bin "alive" (able to fit next item)
    # Scaled by item size to be adaptive
    beta = 1.0 * item
    
    # Smooth probability that the bin can fit the next item
    # sigmoid(remaining - next_item_est)
    # If remaining >> next_item_est, prob ~ 1
    # If remaining << next_item_est, prob ~ 0
    
    # Calculate scores for valid bins
    valid_remaining = remaining[valid]
    valid_scores = -valid_remaining + beta * (1.0 / (1.0 + np.exp(-(valid_remaining - next_item_est))))
    
    scores[valid] = valid_scores
    
    return scores
