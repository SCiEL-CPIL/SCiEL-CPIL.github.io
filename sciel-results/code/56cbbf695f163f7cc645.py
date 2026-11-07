# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix based on Delaunay triangulation.
    Edges in the Delaunay triangulation get high attractiveness; others get low.
    """
    n = distance_matrix.shape[0]
    
    # Extract points from the distance matrix
    # We need to reconstruct points from the distance matrix
    # Since the points are in the unit square, we can use MDS or just assume
    # the distance matrix corresponds to some points.
    
    # Actually, we can't easily get points from distance matrix alone.
    # But wait - the task says "50 uniform random points in the unit square".
    # The distance_matrix is given, but we don't have the points directly.
    
    # Alternative approach: Use the distance matrix to identify "close" neighbors.
    # For each node, find its k nearest neighbors based on distance.
    # Edges between close neighbors get high attractiveness.
    
    # Let's use a k-nearest neighbors approach as a proxy for Delaunay-like structure
    # Since we can't compute Delaunay without points, let's use distance-based heuristic.
    
    # Strategy: For each node i, find the k closest nodes. Mark those edges as attractive.
    # This creates a sparse graph similar to Delaunay triangulation.
    
    k = 8  # Number of nearest neighbors
    
    # Initialize heuristic with a small base value
    heuristic = np.full((n, n), 1e-9)
    
    # Set diagonal to 0 (or small value, but ants won't visit themselves)
    np.fill_diagonal(heuristic, 0.0)
    
    # For each node, find k nearest neighbors
    for i in range(n):
        # Get distances from node i to all others
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        
        # Find k nearest neighbors
        k_nearest = np.argpartition(dists, k)[:k]
        
        # For each nearest neighbor, set high attractiveness
        for j in k_nearest:
            # Attractiveness inversely proportional to distance, but boosted
            # Use a smooth function
            d = distance_matrix[i, j]
            # Higher attractiveness for closer nodes
            attractiveness = 1.0 / (d + 1e-9)
            # Boost for being a nearest neighbor
            attractiveness *= 2.0
            heuristic[i, j] = max(heuristic[i, j], attractiveness)
            heuristic[j, i] = max(heuristic[j, i], attractiveness)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
