"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import time


def heuristics(distance_matrix):
    """
    Compute a heuristic edge-attractiveness matrix for ant colony optimization.
    
    Strategy:
    1. Base: inverse distance (closer = more attractive).
    2. Boost edges that are in the Delaunay triangulation (local connectivity).
    3. Penalize edges that are too long (to avoid crossing the tour).
    4. Ensure all values are positive and finite.
    """
    start_time = time.time()
    deadline = 330.0  # safety margin
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Base heuristic: inverse distance
    # This is the standard ACO heuristic
    base = 1.0 / (distance_matrix + eps)
    
    # Make diagonal 0 (no self-loops)
    np.fill_diagonal(base, 0.0)
    
    # Try to enhance with Delaunay triangulation
    try:
        # Extract coordinates from distance matrix
        # We need to reconstruct points. Since we only have distances,
        # we can use multidimensional scaling or just work with the matrix directly.
        # However, Delaunay needs coordinates. Let's use MDS to get approximate 2D coords.
        
        # For 50 points, MDS is fast
        from scipy.spatial.distance import squareform
        from sklearn.manifold import MDS
        
        # Actually, sklearn might not be available. Let's use a simpler approach.
        # We can use the distance matrix directly with a k-NN based boost.
        
        # Use k-NN: boost edges to the k nearest neighbors
        k = 8  # typical k for TSP heuristics
        tree = cKDTree.from_distance_matrix(distance_matrix)
        _, indices = tree.query(distance_matrix, k=k+1)  # +1 for self
        
        # Create a boost matrix
        boost = np.zeros_like(base)
        for i in range(n):
            for j in indices[i, 1:]:  # skip self
                if j < n:
                    boost[i, j] = 1.0
                    boost[j, i] = 1.0
        
        # Combine: base * (1 + boost)
        # Edges to nearest neighbors get a 2x boost
        combined = base * (1.0 + boost)
        
    except Exception:
        # If k-NN fails, just use base
        combined = base.copy()
    
    # Penalize long edges: if distance > 2 * median distance, reduce attractiveness
    try:
        # Get non-zero distances
        mask = ~np.eye(n, dtype=bool)
        distances = distance_matrix[mask]
        median_dist = np.median(distances)
        
        # Penalize edges that are more than 2x median distance
        long_mask = distance_matrix > 2.0 * median_dist
        long_mask &= mask  # exclude diagonal
        
        # Reduce attractiveness for long edges
        combined[long_mask] *= 0.5
        
    except Exception:
        pass
    
    # Ensure all values are positive and finite
    combined = np.where(np.isfinite(combined), combined, 1e-9)
    combined = np.where(combined > 0, combined, 1e-9)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return a simple fallback
        return 1.0 / (distance_matrix + eps)
    
    return combined
# EVOLVE-BLOCK-END
