# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    t0 = time.time()
    n = distance_matrix.shape[0]
    
    # Work with a copy
    d = distance_matrix.astype(np.float64).copy()
    
    # Set diagonal to infinity for neighbor finding
    np.fill_diagonal(d, np.inf)
    
    # Reconstruct 2D points via classical MDS
    D2 = d ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.clip(eigvals, 0, None)
    coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(coords)
        delaunay_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    delaunay_edges.add((min(a, b), max(a, b)))
    except Exception:
        delaunay_edges = set()
    
    # Find k-nearest neighbors for each node (k=4)
    k = 4
    nn_indices = np.argpartition(d, k, axis=1)[:, :k]
    
    # Build a "nearest-neighbor edge" matrix
    nn_edge = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in nn_indices[i]:
            nn_edge[i, j] = 1.0
            nn_edge[j, i] = 1.0  # symmetrize
    
    # Base heuristic: inverse distance with sharpened exponent
    eps = 1e-9
    d_safe = d + eps
    base = 1.0 / (d_safe ** 2.0)
    
    # Strongly boost nearest-neighbor edges
    nn_boost = 1.0 + 10.0 * nn_edge  # 11x boost for NN edges
    
    # Additional boost for mutual nearest neighbors
    mutual_nn = nn_edge * nn_edge.T
    mutual_boost = 1.0 + 5.0 * mutual_nn
    
    # Combine
    result = base * nn_boost * mutual_boost
    
    # Apply Delaunay boost: edges in the triangulation get an additional boost
    for (a, b) in delaunay_edges:
        result[a, b] *= 3.0
        result[b, a] *= 3.0
    
    # Ensure positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    # Symmetrize
    result = 0.5 * (result + result.T)
    
    # Final safety
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
