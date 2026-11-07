"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger is more attractive).
    
    Strategy: Combine inverse distance with a "nearest-neighbor" bonus.
    Edges that are among the shortest for a given node get a significant boost,
    which helps ants construct locally optimal tours. We also penalize very long
    edges more aggressively by using a power law on the inverse distance.
    
    Specifically:
    1. Compute inverse distance: 1 / d
    2. For each node i, find the k nearest neighbors (k=3).
    3. Add a bonus to edges (i,j) where j is in the k-nearest neighbors of i.
    4. Normalize to ensure positive values and finite values.
    """
    n = distance_matrix.shape[0]
    
    # Avoid division by zero
    d = distance_matrix + 1e-9
    
    # Base heuristic: inverse distance raised to a power to emphasize short edges
    # Power of 2 makes the difference between short and long edges more pronounced
    base_heuristic = 1.0 / (d ** 2)
    
    # Add a nearest-neighbor bonus
    # For each row i, identify the 3 smallest distances
    k = 3
    if n > k:
        # Find indices of k smallest distances for each row
        # argpartition is O(n), sort is O(n log n) but n=50 is small
        k_indices = np.argpartition(d, k, axis=1)[:, :k]
        
        # Create a mask for edges that are in the k-nearest neighbors
        bonus_mask = np.zeros_like(d, dtype=bool)
        for i in range(n):
            for j in k_indices[i]:
                bonus_mask[i, j] = True
        
        # Add a large bonus to these edges to strongly guide the ants
        # The bonus should be comparable to the max base_heuristic
        max_base = np.max(base_heuristic)
        bonus_value = max_base * 0.5
        base_heuristic[bonus_mask] += bonus_value
    
    # Ensure all values are positive and finite
    # Replace any non-positive or non-finite values with a small epsilon
    base_heuristic = np.where(np.isfinite(base_heuristic) & (base_heuristic > 0), base_heuristic, 1e-9)
    
    return base_heuristic
# EVOLVE-BLOCK-END
