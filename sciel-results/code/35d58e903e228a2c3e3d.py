"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of TSP.
    
    Strategy: Use a combination of:
    1. Inverse distance (base heuristic)
    2. Delaunay triangulation-based local structure bonus
    3. Angle-based penalty (prefer smoother paths)
    
    The heuristic should be larger for more attractive edges.
    """
    n = distance_matrix.shape[0]
    
    # Base: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Extract coordinates from distance matrix using multilateration
    # For uniform random points in unit square, we can try to recover coords
    # Actually, we only have the distance matrix, so let's work with it directly.
    
    # Approach: Use the distance matrix to identify "local" neighbors
    # Edges to nearby nodes should be more attractive
    
    # Compute rank-based local structure
    # For each node, find its k-nearest neighbors
    k = 5
    heuristic = np.zeros_like(distance_matrix)
    
    for i in range(n):
        # Get distances from node i to all others
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        
        # Find k nearest neighbors
        k_indices = np.argpartition(dists, k)[:k]
        
        # For these k nearest neighbors, boost the heuristic
        for j in k_indices:
            # Use a smooth function that gives higher weight to closer neighbors
            # among the k nearest
            heuristic[i, j] += 1.0 / (distance_matrix[i, j] + 1e-9)
    
    # Normalize by number of times each edge was boosted (each edge appears in 
    # at most 2 nodes' k-NN lists)
    # Actually, let's just use the sum as is, it naturally gives more weight to 
    # edges that are k-nearest for both endpoints
    
    # Add a small component based on inverse distance squared to emphasize locality
    locality_bonus = 1.0 / (distance_matrix + 1e-9) ** 2
    locality_bonus = locality_bonus / locality_bonus.max() * 0.5
    
    # Combine: primary signal from k-NN structure, secondary from inverse distance
    # Normalize k-NN heuristic
    k_nn_norm = heuristic / heuristic.max() if heuristic.max() > 0 else np.zeros_like(heuristic)
    
    # Final heuristic: weighted combination
    result = 0.6 * k_nn_norm + 0.4 * locality_bonus
    
    # Ensure all values are positive
    result = np.maximum(result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
