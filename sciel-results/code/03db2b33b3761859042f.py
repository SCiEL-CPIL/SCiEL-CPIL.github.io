"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix based on the rank of neighbors.
    For each node i, the nearest neighbor j gets the highest heuristic value,
    and the values decay as the rank increases.
    This emphasizes local connectivity which is crucial for TSP tours.
    """
    n = distance_matrix.shape[0]
    # We need to compute ranks for each row.
    # rank 1 is the closest (excluding self, though self is 0 distance usually).
    # The problem states "50 uniform random points", so distance_matrix[i][i] is 0.
    # We should ignore self-loops in the ranking or treat them as rank 0/infinite.
    
    # Create a copy to handle the diagonal
    d = distance_matrix.copy()
    
    # Set diagonal to infinity so they don't interfere with ranking of actual neighbors
    np.fill_diagonal(d, np.inf)
    
    # For each row, get the indices that would sort the distances
    # rank[i][j] will be 1 for the closest, 2 for the second closest, etc.
    ranks = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        # argsort gives indices that would sort the row
        sorted_indices = np.argsort(d[i])
        # Assign ranks: 1, 2, ..., n-1
        # sorted_indices[0] is the closest, so it gets rank 1
        for rank_idx, j in enumerate(sorted_indices):
            ranks[i][j] = rank_idx + 1
            
    # Heuristic: 1 / (rank + epsilon)
    # Using a small epsilon to avoid division by zero if any weirdness, though ranks start at 1.
    # The diagonal (self) has rank 0 in our loop? No, we set d[i][i] to inf, so it will be last in sort.
    # Its rank will be n-1 (index n-2 + 1). 
    # We want the diagonal to be 0 or very low because ants shouldn't go to themselves.
    # The ACO rule says "Heuristic entries that are not positive become 1e-9".
    # So we can explicitly set the diagonal to 0.
    
    # Construct the heuristic matrix
    eta = 1.0 / (ranks + 1.0)
    
    # Ensure diagonal is 0 (or negative, which maps to 1e-9 in ACO, but 0 is cleaner)
    np.fill_diagonal(eta, 0.0)
    
    return eta
# EVOLVE-BLOCK-END
