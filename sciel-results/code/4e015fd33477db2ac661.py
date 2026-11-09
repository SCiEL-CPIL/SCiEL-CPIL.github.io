"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ACO bin packing.
    
    Strategy: Use a First-Fit-Decreasing (FFD) inspired heuristic that rewards
    placing items that fit together well. We combine:
    1. A base fit score: how well the pair fills the bin (closer to capacity is better)
    2. A size similarity bonus: items of similar size tend to pack well together
    3. A large-item priority: placing large items early is important
    
    The heuristic is computed once per instance.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Pairwise sum of sizes
    pair_sum = demands[:, None] + demands[None, :]
    
    # Only valid if pair fits in capacity
    valid = pair_sum <= capacity
    
    # Base heuristic: reward pairs that fill the bin well (closer to capacity)
    # Use a smooth function: higher when pair_sum is close to capacity but not over
    # Normalize: pair_sum / capacity ranges from ~0.27 to 1.0 for valid pairs
    fill_ratio = np.where(valid, pair_sum / capacity, 0.0)
    
    # Reward being close to full: use (fill_ratio)^2 to emphasize full bins
    # But also penalize very small pairs
    base_score = np.where(valid, fill_ratio ** 2, 0.0)
    
    # Size similarity bonus: items of similar size pack well
    # Use 1 - |d_i - d_j| / capacity as a similarity measure
    size_diff = np.abs(demands[:, None] - demands[None, :]) / capacity
    similarity = np.where(valid, 1.0 - size_diff, 0.0)
    
    # Large item priority: if either item is large, give a boost
    # Large items are harder to place, so placing them with good partners is key
    max_item = np.maximum(demands[:, None], demands[None, :])
    large_boost = np.where(valid, max_item / capacity, 0.0)
    
    # Combine: weight base heavily, add similarity and large item boost
    # This encourages: filling bins well, grouping similar sizes, and handling large items
    heuristic = 2.0 * base_score + 1.0 * similarity + 0.5 * large_boost
    
    # Ensure non-positive values become 1e-9
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
