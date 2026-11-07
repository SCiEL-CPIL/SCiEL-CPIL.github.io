# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix based on Delaunay triangulation.
    Uses a topological adjacency heuristic where only edges in the Delaunay
    triangulation are considered highly attractive.
    """
    n = distance_matrix.shape[0]
    
    # 1. Reconstruct 2D points from the distance matrix using Classical MDS
    D = distance_matrix
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 eigenvalues for 2D embedding
    # Clip small negative eigenvalues to 0 to handle numerical noise
    eigenvalues[0:2] = np.maximum(eigenvalues[0:2], 0)
    
    # Compute 2D coordinates
    points = eigenvectors[:, 0:2] * np.sqrt(eigenvalues[0:2])
    
    # 2. Compute Delaunay Triangulation
    try:
        tri = Delaunay(points)
        simplices = tri.simplices
    except Exception:
        # Fallback: If Delaunay fails (e.g., all points collinear or degenerate),
        # use k-nearest neighbors as a sparse graph approximation
        k = 6
        heuristic = np.full((n, n), 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        for i in range(n):
            dists = distance_matrix[i].copy()
            dists[i] = np.inf
            k_nearest = np.argpartition(dists, k)[:k]
            for j in k_nearest:
                d = distance_matrix[i, j]
                # Use inverse square distance for stronger preference for short edges
                attractiveness = 1.0 / (d ** 2 + 1e-12)
                heuristic[i, j] = max(heuristic[i, j], attractiveness)
                heuristic[j, i] = max(heuristic[j, i], attractiveness)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic

    # 3. Build Heuristic Matrix
    # Initialize with a very low value (noise floor)
    heuristic = np.full((n, n), 1e-9)
    np.fill_diagonal(heuristic, 0.0)
    
    # Assign high attractiveness to edges in the Delaunay triangulation
    # We use inverse square distance to strongly favor shorter edges within the valid subgraph
    for simplex in simplices:
        i, j, k = simplex
        edges = [(i, j), (j, k), (i, k)]
        for a, b in edges:
            d = distance_matrix[a, b]
            # Inverse square law: emphasizes short edges more than 1/d
            attractiveness = 1.0 / (d ** 2 + 1e-12)
            heuristic[a, b] = max(heuristic[a, b], attractiveness)
            heuristic[b, a] = max(heuristic[b, a], attractiveness)
            
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
