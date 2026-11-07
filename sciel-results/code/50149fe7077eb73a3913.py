"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO using Delaunay triangulation
    with a demand-aware weighting scheme.
    """
    n = len(demands)
    
    # Get customer indices (excluding depot at index 0)
    customer_indices = np.arange(1, n)
    customer_coords = coordinates[customer_indices]
    
    try:
        # Perform Delaunay triangulation on customer coordinates
        tri = Delaunay(customer_coords)
        
        # Build adjacency from triangulation
        adjacency = np.zeros((n, n), dtype=bool)
        
        # For each triangle, add edges between connected customers
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    ci = simplex[i] + 1  # Convert to original index
                    cj = simplex[j] + 1  # Convert to original index
                    adjacency[ci, cj] = True
                    adjacency[cj, ci] = True
        
        # Connect depot to its nearest neighbors
        depot_distances = distance_matrix[0, 1:n]
        k_nearest = min(12, n - 1)
        nearest_indices = np.argpartition(depot_distances, k_nearest)[:k_nearest] + 1
        for idx in nearest_indices:
            adjacency[0, idx] = True
            adjacency[idx, 0] = True
        
        # Create heuristic matrix
        heuristic = np.full((n, n), 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        
        # Compute base heuristic using inverse distance for Delaunay edges
        # with demand-aware scaling
        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j]:
                    dist = distance_matrix[i, j]
                    if dist > 0:
                        # Base inverse distance
                        base = 1.0 / (dist + 1e-9)
                        
                        # Demand-aware scaling: edges between low-demand nodes
                        # are slightly more attractive (easier to fit in route)
                        di = demands[i] if i > 0 else 0
                        dj = demands[j] if j > 0 else 0
                        demand_factor = 1.0 + 0.1 * (capacity - di - dj) / capacity
                        heuristic[i, j] = base * demand_factor
        
        # Ensure no zero or negative values (except diagonal)
        mask = ~np.eye(n, dtype=bool)
        heuristic[mask] = np.maximum(heuristic[mask], 1e-9)
        
    except Exception:
        # Fallback to simple inverse distance heuristic
        heuristic = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        mask = ~np.eye(n, dtype=bool)
        heuristic[mask] = np.maximum(heuristic[mask], 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
