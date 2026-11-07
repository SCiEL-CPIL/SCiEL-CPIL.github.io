"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO using enhanced Delaunay triangulation.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n, n)
    coordinates : np.ndarray
        Coordinates of shape (n, 2)
    demands : np.ndarray
        Demands of shape (n,)
    capacity : float
        Vehicle capacity
    
    Returns:
    --------
    np.ndarray
        Heuristic matrix of same shape as distance_matrix
    """
    n = len(demands)
    
    # Get customer indices (excluding depot at index 0)
    customer_indices = np.arange(1, n)
    customer_coords = coordinates[customer_indices]
    
    # Perform Delaunay triangulation on customer coordinates
    try:
        tri = Delaunay(customer_coords)
        
        # Build adjacency from triangulation
        adjacency = np.zeros((n, n), dtype=bool)
        
        # For each triangle, add edges between connected customers
        for simplex in tri.simplices:
            # simplex contains indices into customer_coords (0 to n-2)
            for i in range(3):
                for j in range(i + 1, 3):
                    ci = simplex[i] + 1  # Convert to original index
                    cj = simplex[j] + 1  # Convert to original index
                    adjacency[ci, cj] = True
                    adjacency[cj, ci] = True
        
        # Add k-nearest neighbors for better connectivity
        # Use k=5 nearest neighbors for each customer
        k = 5
        for i in range(1, n):
            dists_to_others = distance_matrix[i, 1:n].copy()
            dists_to_others[i-1] = np.inf  # Exclude self
            k_nearest = np.argpartition(dists_to_others, k)[:k] + 1
            for j in k_nearest:
                adjacency[i, j] = True
                adjacency[j, i] = True
        
        # Connect depot to its nearest neighbors more aggressively
        depot_distances = distance_matrix[0, 1:n]
        # Connect depot to top 15 nearest customers
        k_depot = min(15, n - 1)
        nearest_indices = np.argpartition(depot_distances, k_depot)[:k_depot] + 1
        for idx in nearest_indices:
            adjacency[0, idx] = True
            adjacency[idx, 0] = True
        
        # Create heuristic matrix with distance-aware weighting
        heuristic = np.zeros((n, n), dtype=float)
        
        # For edges in the enhanced graph, use a weighted inverse distance
        # Weight by how short the edge is relative to the median edge length
        edge_lengths = []
        for i in range(n):
            for j in range(i+1, n):
                if adjacency[i, j] and distance_matrix[i, j] > 0:
                    edge_lengths.append(distance_matrix[i, j])
        
        if len(edge_lengths) > 0:
            median_dist = np.median(edge_lengths)
            max_dist = np.max(edge_lengths)
            
            for i in range(n):
                for j in range(n):
                    if i != j and adjacency[i, j]:
                        dist = distance_matrix[i, j]
                        if dist > 0:
                            # Use a power-law weighting: shorter edges get much higher weight
                            # Normalize distance to [0, 1] and apply power
                            norm_dist = dist / (max_dist + 1e-9)
                            # Weight is inversely proportional, with stronger penalty for long edges
                            weight = (1.0 - norm_dist) ** 2
                            heuristic[i, j] = weight / (dist + 1e-9)
        
        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(heuristic, 0.0)
        
        # Ensure no zero or negative values (except diagonal) - use 1e-9 for non-graph edges
        mask = ~np.eye(n, dtype=bool)
        heuristic[mask] = np.maximum(heuristic[mask], 1e-9)
        
    except Exception:
        # Fallback to simple inverse distance heuristic
        heuristic = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        heuristic[~np.eye(n, dtype=bool)] = np.maximum(heuristic[~np.eye(n, dtype=bool)], 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
