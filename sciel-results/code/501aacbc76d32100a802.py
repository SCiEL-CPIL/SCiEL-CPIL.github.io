"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    if n < 3:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Reconstruct point coordinates from distance matrix (up to rigid transform)
    # Use the distance matrix to recover 2D coordinates via classical MDS
    # Since points are in the unit square, we can use the distance matrix directly
    
    # Compute coordinates using classical MDS (double centering)
    D2 = distance_matrix ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take top 2 dimensions
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(coords)
        simplices = tri.simplices
    except Exception:
        # Fallback: use k-NN based heuristic
        k = min(6, n - 1)
        heuristic = np.zeros((n, n))
        for i in range(n):
            dists = distance_matrix[i].copy()
            dists[i] = np.inf
            knn_indices = np.argsort(dists)[:k]
            for j in knn_indices:
                heuristic[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
            # Also add reverse
            for j in knn_indices:
                heuristic[j, i] = 1.0 / (distance_matrix[j, i] + 1e-9)
        # Normalize
        heuristic = np.maximum(heuristic, 1e-9)
        return heuristic
    
    # Build adjacency from Delaunay triangulation
    delaunay_adj = np.zeros((n, n), dtype=np.float64)
    for simplex in simplices:
        for a in range(3):
            for b in range(3):
                if a != b:
                    delaunay_adj[simplex[a], simplex[b]] = 1.0
    
    # Compute heuristic: high for Delaunay edges, low for others
    # Use a combination: Delaunay edges get boosted inverse distance,
    # non-Delaunay edges get heavily suppressed inverse distance
    heuristic = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 1e-9
                continue
            d = distance_matrix[i, j]
            if delaunay_adj[i, j] > 0:
                # Delaunay edge: high attractiveness, inverse distance
                heuristic[i, j] = 1.0 / (d + 1e-9)
            else:
                # Non-Delaunay edge: heavily suppressed
                # Use a strong decay to make these edges very unattractive
                heuristic[i, j] = 1e-4 / (d + 1e-9)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
