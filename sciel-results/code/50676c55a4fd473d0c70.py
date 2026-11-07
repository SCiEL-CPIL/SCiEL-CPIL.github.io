"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses a topological k-Nearest Neighbors heuristic derived from the 
    approximation of the Delaunay triangulation structure.
    Edges within the symmetric k-NN graph are assigned high attractiveness 
    (inverse distance), while all other edges are assigned minimal 
    attractiveness (1e-9). This prunes the search space to a sparse, 
    geometrically valid subgraph likely to contain the optimal tour.
    """
    N = distance_matrix.shape[0]
    
    # Work with a copy to avoid modifying input
    dist = distance_matrix.astype(np.float64).copy()
    
    # Ensure diagonal is 0 for distance calculations
    np.fill_diagonal(dist, 0.0)
    
    # Define k for nearest neighbors.
    # k=6 is the average degree of a Delaunay triangulation in 2D.
    # Using a symmetric k-NN graph approximates the Delaunay adjacency.
    k = 6
    
    # Find the indices of the k nearest neighbors for each node
    # argpartition is O(N) on average, but for N=50, argsort is fine and simpler
    # We need the indices of the k smallest values in each row.
    # np.argsort(dist, axis=1)[:, :k] gives the indices of the k smallest distances.
    knn_indices = np.argsort(dist, axis=1)[:, :k]
    
    # Create a boolean mask for the symmetric k-NN graph
    # is_knn[i, j] is True if j is in kNN(i) or i is in kNN(j)
    is_knn = np.zeros((N, N), dtype=bool)
    for i in range(N):
        is_knn[i, knn_indices[i]] = True
    
    # Symmetrize: if i is in kNN(j), then is_knn[i, j] should be True
    # This is already handled by the loop if we iterate all i, but let's be explicit
    # is_knn = is_knn | is_knn.T
    
    # Calculate base heuristic: 1 / distance
    # Avoid division by zero on diagonal
    safe_dist = np.where(dist > 1e-10, dist, 1.0)
    base_heuristic = 1.0 / safe_dist
    
    # Apply the topological mask
    # High attractiveness for kNN edges, very low for others
    # 1e-9 is the floor specified in the task for non-positive entries, 
    # but we use it as a low value for non-topological edges.
    low_value = 1e-9
    
    heuristic = np.where(is_knn, base_heuristic, low_value)
    
    # Set diagonal to 0 (or very small) as ants don't move to themselves
    np.fill_diagonal(heuristic, 0.0)
    
    # Final cleanup for any non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
