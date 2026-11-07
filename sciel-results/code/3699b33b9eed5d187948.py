# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness heuristic matrix for ant colony optimization
    of the TSP using a rank-based mutual proximity metric.
    
    This approach assigns higher attractiveness to edges where both nodes
    rank each other as close neighbors. This is robust to local density
    variations and emphasizes the local topology of the optimal tour.
    """
    # Ensure we're working with a numpy array
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    if n <= 1:
        return np.ones_like(dm)
        
    # We need to compute ranks for each row.
    # Set diagonal to infinity so it gets the worst rank.
    dm_ranks = np.zeros_like(dm, dtype=np.float64)
    
    for i in range(n):
        # Get distances from node i to all others
        dists = dm[i].copy()
        dists[i] = np.inf
        
        # Get the indices that would sort the distances
        # np.argsort returns indices that would sort the array
        # We want rank: smallest distance -> rank 1
        sorted_indices = np.argsort(dists)
        
        # Initialize ranks for this row
        ranks = np.zeros(n, dtype=np.float64)
        # Assign ranks 1 to n based on sorted order
        # sorted_indices[k] is the node with the k-th smallest distance (0-indexed)
        # So its rank is k+1
        for rank_val, node_idx in enumerate(sorted_indices, start=1):
            ranks[node_idx] = rank_val
            
        # The diagonal element (node i) should have a high rank or be handled separately
        # Since we set dists[i]=inf, it should naturally get the highest rank (n)
        # But let's be explicit just in case of ties or numerical issues
        ranks[i] = n
        
        dm_ranks[i] = ranks
        
    # Now dm_ranks[i, j] is the rank of j from i's perspective.
    # We want a symmetric heuristic based on mutual ranking.
    # H_ij = 1 / (rank_i(j) + rank_j(i))
    
    # To avoid division by zero (though ranks are >= 1, sum >= 2), we add a small epsilon if needed, but it's not necessary.
    # However, we must handle the diagonal.
    # For diagonal, rank is n. Sum = 2n. H_ii = 1/(2n).
    # This is fine, ants won't pick self-loops anyway (unvisited constraint).
    
    heuristic = 1.0 / (dm_ranks + dm_ranks.T)
    
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
