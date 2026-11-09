import numpy as np

def heuristics(prizes, weights):
    # Aggregate prizes to one value per item
    prize_per_item = prizes.mean(axis=0)  # shape (100,)
    
    # Compute weight metrics per item
    max_w = weights.max(axis=0)   # shape (100,)
    mean_w = weights.mean(axis=0) # shape (100,)
    
    # Blend: emphasize the most constraining dimension
    denom = 0.7 * max_w + 0.3 * mean_w + 1e-6
    
    scores = prize_per_item / denom
    
    # Ensure positive
    scores = np.where(scores > 0, scores, 1e-9)
    
    return scores
