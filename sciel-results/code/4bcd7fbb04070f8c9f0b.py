"""Heuristic matrix for ACO-TSP: MDS + Delaunay + Convex Hull boosting."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
import sys

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 5s safety margin

def _mds_coordinates(D):
    """
    Recover 2D coordinates from a Euclidean distance matrix using Classical MDS.
    Returns coordinates or None if it fails.
    """
    n = D.shape[0]
    if n < 3:
        return None
    
    # Check for symmetry and non-negativity
    if not np.allclose(D, D.T):
        return None
    if np.any(D < -1e-9):
        return None

    # Classical MDS
    # B = -0.5 * J * D^2 * J
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    
    # Symmetrize B to remove numerical noise
    B = (B + B.T) / 2.0
    
    try:
        w, V = np.linalg.eigh(B)
    except:
        return None
        
    # Sort eigenvalues in descending order
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    
    # The first two eigenvalues should be positive for 2D points
    if w[0] < 1e-9:
        return None
        
    # Coordinates
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    
    # Relax tolerance check to avoid false negatives from numerical noise
    # If points are truly Euclidean, error should be tiny.
    # We'll just check if the reconstruction is "reasonable" (correlation > 0.99)
    from scipy.spatial.distance import pdist, squareform
    rec_D = squareform(pdist(coords))
    
    # Check correlation
    if np.corrcoef(rec_D.flatten(), D.flatten())[0, 1] < 0.99:
        return None
        
    return coords

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Recover coordinates via MDS.
    2. Compute Delaunay Triangulation and Convex Hull.
    3. Base heuristic: Inverse 4th power of distance.
    4. Boost Delaunay edges.
    5. Boost Convex Hull edges.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        # Trivial case
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Base Heuristic: Inverse 2nd power (less aggressive than 4th)
    # Add small epsilon to diagonal to avoid inf, then set diagonal to 0 later
    heuristic = 1.0 / (d + 1e-10) ** 2
    
    # 2. K-Nearest Neighbors Boost
    # Boost edges to the 5 nearest neighbors for each node
    k = 5
    knn_boost = 2.0
    # d has 0 on diagonal (or very small), so we can sort rows
    # Note: d is symmetric
    for i in range(n):
        # Get indices of k nearest neighbors
        # Exclude self (index i)
        dists = d[i].copy()
        dists[i] = np.inf
        nn_indices = np.argpartition(dists, k)[:k]
        for j in nn_indices:
            heuristic[i, j] *= knn_boost
            heuristic[j, i] *= knn_boost

    # 3. Try to recover coordinates and use geometry
    coords = _mds_coordinates(d)
    
    if coords is not None:
        try:
            # Delaunay Triangulation
            tri = scipy.spatial.Delaunay(coords)
            
            # Boost Delaunay edges
            delaunay_boost = 1.5
            for simplex in tri.simplices:
                # A simplex in 2D is a triangle (i, j, k)
                # Edges are (i,j), (j,k), (i,k)
                i, j, k = simplex
                heuristic[i, j] *= delaunay_boost
                heuristic[j, i] *= delaunay_boost
                heuristic[j, k] *= delaunay_boost
                heuristic[k, j] *= delaunay_boost
                heuristic[i, k] *= delaunay_boost
                heuristic[k, i] *= delaunay_boost
                
            # Convex Hull
            hull = scipy.spatial.ConvexHull(coords)
            
            # Boost Hull edges
            hull_boost = 3.0
            for simplex in hull.simplices:
                # In 2D, hull.simplices are pairs of indices
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
                
        except Exception:
            # If geometry fails, we still have the base heuristic
            pass

    # Ensure diagonal is 0 (or very small)
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check for finiteness and positivity
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
