"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import time


def _mst_heuristic(distance_matrix):
    """
    Compute a heuristic matrix that emphasizes MST edges.
    The heuristic is 1/distance for non-MST edges, and a boosted value for MST edges.
    This provides a global structural guide for the ant colony.
    """
    n = distance_matrix.shape[0]
    
    # Create a graph from the distance matrix for MST computation
    # We need to handle the fact that distance_matrix has 0 on the diagonal
    # and we want to find the MST of the complete graph with edge weights = distances
    
    # Convert to a sparse graph for MST
    # Use scipy's minimum_spanning_tree which works on dense matrices too
    # but let's use the sparse version for clarity
    
    # Create a directed graph representation (MST is undirected, but we need to handle both directions)
    # Actually, minimum_spanning_tree expects a square matrix where entry [i,j] is the weight
    # from i to j. For an undirected graph, we'd have symmetric weights.
    
    # The distance matrix is already symmetric for Euclidean distances
    mst_matrix = minimum_spanning_tree(distance_matrix).toarray()
    
    # MST edges have positive weights in mst_matrix, non-MST edges have inf
    # Create a mask for MST edges
    mst_mask = mst_matrix < np.inf
    
    # Base heuristic: inverse distance
    # Handle zero distances (diagonal) by using a small epsilon
    dist_safe = np.where(distance_matrix > 1e-9, distance_matrix, 1e-9)
    base_heuristic = 1.0 / dist_safe
    
    # Boost MST edges
    # The boost factor can be tuned; let's use a moderate boost
    boost_factor = 2.0
    heuristic = base_heuristic.copy()
    heuristic[mst_mask] *= boost_factor
    
    # Ensure diagonal is handled (should be 0 distance, so 1/1e-9 which is large, but we don't use diagonal)
    # Actually, for the diagonal, distance is 0, so dist_safe is 1e-9, heuristic is 1e9
    # But we never traverse to the same node, so it doesn't matter much
    # However, let's set diagonal to 0 to be safe
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ant colony optimization.
    Uses MST-based heuristic to provide global structural guidance.
    """
    start_time = time.time()
    
    # Set a deadline with safety margin (330s limit, use 300s to be safe)
    deadline = start_time + 280.0
    
    try:
        # Compute the MST-based heuristic
        heuristic = _mst_heuristic(distance_matrix)
        
        # Check if we're still within deadline
        if time.time() > deadline:
            # Fall back to simple inverse distance if MST took too long
            dist_safe = np.where(distance_matrix > 1e-9, distance_matrix, 1e-9)
            heuristic = 1.0 / dist_safe
            np.fill_diagonal(heuristic, 0.0)
            heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        
        return heuristic
        
    except Exception:
        # Fallback to simple inverse distance on any error
        dist_safe = np.where(distance_matrix > 1e-9, distance_matrix, 1e-9)
        heuristic = 1.0 / dist_safe
        np.fill_diagonal(heuristic, 0.0)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
# EVOLVE-BLOCK-END
