"""Heuristic matrix for ant colony optimization of the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix using the MST structure.
    
    Strategy: Build the MST of the distance matrix, compute shortest path distances
    within the MST, and define the heuristic as exp(-alpha * mst_shortest_path_length).
    This penalizes edges that would create cycles with long detours in the tree structure.
    """
    n = distance_matrix.shape[0]
    
    # Build the MST using scipy
    # minimum_spanning_tree expects a weighted graph; we pass the distance matrix
    mst = minimum_spanning_tree(distance_matrix)
    
    # Convert to dense matrix for shortest path computation
    mst_dense = mst.toarray()
    
    # Replace zeros (except diagonal) with infinity to represent non-edges
    # Actually, minimum_spanning_tree already gives 0 for non-edges and weights for edges
    # We need to handle this properly
    mst_dense = np.where(mst_dense > 0, mst_dense, np.inf)
    np.fill_diagonal(mst_dense, 0)
    
    # Compute all-pairs shortest paths within the MST
    mst_sp = shortest_path(mst_dense, method='D')
    
    # The heuristic is exp(-alpha * mst_sp), where alpha controls the strength
    # We want larger values to be more attractive, so shorter MST paths get higher heuristic
    alpha = 1.0
    
    # Avoid issues with inf values
    mst_sp_safe = np.where(np.isfinite(mst_sp), mst_sp, 1e10)
    
    heuristic = np.exp(-alpha * mst_sp_safe)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
