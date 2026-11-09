import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores

    valid_remaining = remaining[valid]
    
    # Base score: Best Fit (minimize remaining)
    base_score = -valid_remaining
    
    # Fragmentation penalty
    # We penalize remaining space that is "moderate" relative to the item.
    # If remaining is small, it's a good fit.
    # If remaining is large, it's a safe fit.
    # The "bad" zone is where remaining is neither very small nor very large.
    
    # Scale the "bad zone" with the item size
    lower_bound = item * 0.5
    upper_bound = item * 2.5
    
    # Penalty is a Gaussian bump centered in the bad zone
    center = (lower_bound + upper_bound) / 2
    width = (upper_bound - lower_bound) / 2
    
    # Only apply penalty if remaining is within or near the bad zone
    # z-score for the Gaussian
    z = (valid_remaining - center) / width
    penalty = 10.0 * np.exp(-z**2)
    
    # We want to MINIMIZE the score in the bad zone, so we SUBTRACT the penalty?
    # Wait, higher score is better.
    # Base score is -remaining.
    # If remaining is in the bad zone, we want the score to be LOWER.
    # So we subtract the penalty.
    
    valid_scores = base_score - penalty
    
    scores[valid] = valid_scores
    return scores
