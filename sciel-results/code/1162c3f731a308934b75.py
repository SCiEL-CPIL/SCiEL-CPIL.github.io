"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle potential shape issues, though contract says (5, 100)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # prizes: (5, 100), weights: (5, 100)
    
    # 1. Calculate total prize per item
    # Sum prizes across all 5 constraints
    prize_sum = np.sum(prizes, axis=0) # shape (100,)
    
    # 2. Normalize weights per constraint
    # For each constraint c, divide weights by the max weight in that constraint.
    # This scales each constraint's weights to [0, 1] (assuming max > 0).
    # max_w_c is shape (5,)
    max_w_c = np.max(weights, axis=1, keepdims=True)
    
    # Avoid division by zero
    max_w_c_safe = np.where(max_w_c > 0, max_w_c, 1.0)
    
    # Normalized weights: shape (5, 100)
    norm_weights = weights / max_w_c_safe
    
    # 3. Calculate the bottleneck normalized weight per item
    # This is the maximum normalized weight across all constraints for each item.
    # This represents the item's "cost" in the tightest relative constraint.
    bottleneck_weight = np.max(norm_weights, axis=0) # shape (100,)
    
    # 4. Compute raw efficiency score
    # We want to maximize prize_sum / bottleneck_weight.
    # Add epsilon to avoid division by zero.
    eps = 1e-9
    raw_scores = prize_sum / (bottleneck_weight + eps)
    
    # 5. Handle non-finite values
    raw_scores = np.where(np.isfinite(raw_scores), raw_scores, 0.0)
    
    # 6. Ensure positivity and scale
    # The ACO algorithm uses pheromone * heuristic.
    # Heuristic values should be positive.
    # We can keep the raw magnitude or scale it.
    # Scaling to [1, N] or similar can help, but raw ratios are often fine.
    # Let's shift and scale slightly to ensure a good range, or just ensure positive.
    
    # If all scores are 0, return uniform 1.0
    if np.all(raw_scores == 0):
        return np.ones_like(raw_scores)
    
    # Ensure all scores are positive
    raw_scores = np.maximum(raw_scores, 1e-9)
    
    # Optional: Normalize heuristic to mean 1 or max 1 to prevent pheromone dominance issues?
    # The task says "larger is more attractive".
    # Often, heuristics are scaled so that the best is 1.0 or similar.
    # Let's normalize so that the maximum heuristic is 1.0, and others are proportional.
    # This keeps the values in a stable range [0, 1].
    max_score = np.max(raw_scores)
    if max_score > 0:
        heuristic_scores = raw_scores / max_score
    else:
        heuristic_scores = np.ones_like(raw_scores)
        
    # Ensure strictly positive for ACO probability calculation
    heuristic_scores = np.maximum(heuristic_scores, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        pass # Return whatever we have
        
    return heuristic_scores
# EVOLVE-BLOCK-END
