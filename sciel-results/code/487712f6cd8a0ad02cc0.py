# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness heuristic matrix for ant colony optimization
    of the TSP using a k-Nearest Neighbors (k-NN) geometric filter.
    
    This approach identifies edges that are part of the local neighborhood
    structure (k-NN graph) and boosts their attractiveness using inverse
    distance, while penalizing non-local edges. This focuses the search
    on likely tour edges and suppresses long, suboptimal jumps.
    """
    # Ensure we're working with a numpy array
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    if n <= 1:
        return np.ones_like(dm)
        
    # Work on a copy to handle diagonal
    dm_work = dm.copy()
    np.fill_diagonal(dm_work, np.inf)
    
    # Base heuristic: Inverse distance
    # This captures the geometric cost directly
    with np.errstate(divide='ignore', invalid='ignore'):
        base_heuristic = 1.0 / dm_work
        
    # Define k-NN structure
    # Each node in a TSP tour has exactly 2 neighbors.
    # We consider an edge "strong" if it is among the k nearest neighbors
    # for either endpoint. k=2 is a tight fit for TSP.
    k = 2
    
    # Create a boolean mask for strong edges
    strong_mask = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        # Find indices of the k smallest distances in row i
        # argpartition is faster than argsort for this purpose
        # We get the k smallest values, but we need their indices
        # Note: argpartition returns indices of the k smallest elements, 
        # not necessarily sorted, which is fine for a mask.
        try:
            # Handle case where n might be small or k >= n
            k_eff = min(k, n - 1)
            if k_eff > 0:
                idxs = np.argpartition(dm_work[i], k_eff - 1)[:k_eff]
                strong_mask[i, idxs] = True
        except Exception:
            pass

    # Symmetrize the mask: an edge is strong if i->j is strong OR j->i is strong
    strong_mask = strong_mask | strong_mask.T
    
    # Apply the heuristic
    # Strong edges: use base inverse distance
    # Weak edges: use base inverse distance * penalty
    # The penalty suppresses long edges that are not locally close.
    penalty_factor = 0.1
    
    final_heuristic = base_heuristic.copy()
    final_heuristic[~strong_mask] *= penalty_factor
    
    # Handle diagonal: set to a small positive value to satisfy constraints
    # Ants cannot visit visited nodes, so diagonal is never picked,
    # but it must be finite and positive.
    np.fill_diagonal(final_heuristic, 1e-9)
    
    # Ensure positivity and finiteness
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
