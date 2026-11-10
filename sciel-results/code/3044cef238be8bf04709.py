# EVOLVE-BLOCK-START
import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Returns the set of 220 integer points in 11D with squared norm 2.
    This set satisfies min_pairwise_dist_sq (2) >= max_norm_sq (2).
    """
    d = 11
    # Generate all pairs (i, j) with i < j
    indices = [(i, j) for i in range(d) for j in range(i + 1, d)]
    
    # For each pair, generate 4 sign variations
    # We can construct the array directly for speed
    num_points = len(indices) * 4
    points = np.zeros((num_points, d), dtype=np.int64)
    
    idx = 0
    for i, j in indices:
        # (+, +)
        points[idx, i] = 1
        points[idx, j] = 1
        idx += 1
        # (+, -)
        points[idx, i] = 1
        points[idx, j] = -1
        idx += 1
        # (-, +)
        points[idx, i] = -1
        points[idx, j] = 1
        idx += 1
        # (-, -)
        points[idx, i] = -1
        points[idx, j] = -1
        idx += 1
        
    return points

# EVOLVE-BLOCK-END
