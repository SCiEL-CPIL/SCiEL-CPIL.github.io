"""Rewritten program: Delaunay-triangulation-guided heuristic for ACO TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix for the ACO heuristic.
    
    Strategy: Use Delaunay triangulation to identify geometrically valid edges
    (the optimal TSP tour for points in the plane is a subset of the Delaunay
    triangulation edges plus some additional edges). We boost Delaunay edges
    strongly and penalize non-Delaunay edges, while still allowing them
    to be traversed. This dramatically reduces the effective search space
    and guides ants toward geometrically plausible tours.
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return distance_matrix.copy()
    
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        return 1.0 / (distance_matrix + 1e-9)
    
    eps = 1e-9
    start_time = time.time()
    
    # Base heuristic: inverse distance
    base = 1.0 / (distance_matrix + eps)
    np.fill_diagonal(base, 0.0)
    
    if n <= 6:
        # For very small instances, just use inverse distance
        return np.where(base > 0, base, 1e-9)
    
    # Try to reconstruct point coordinates from the distance matrix
    # Using classical multidimensional scaling (CMDS)
    points = _reconstruct_points(distance_matrix)
    
    if points is None or points.shape[0] != n:
        # Fallback: use pure inverse distance with k-NN boost
        return _knn_heuristic(distance_matrix, base, eps)
    
    # Compute Delaunay triangulation
    delaunay_edges = _delaunay_edge_set(points, n)
    
    if delaunay_edges is None or len(delaunay_edges) == 0:
        return _knn_heuristic(distance_matrix, base, eps)
    
    # Build heuristic:
    # - Delaunay edges: strong inverse distance (high attractiveness)
    # - Non-Delaunay edges: weak inverse distance (low attractiveness)
    # This creates a strong bias toward geometrically valid edges
    
    heuristic = np.zeros((n, n), dtype=float)
    
    # For non-Delaunay edges, use a weak base
    heuristic = base * 0.01  # Very weak for all edges initially
    
    # For Delaunay edges, use strong inverse distance
    for (i, j) in delaunay_edges:
        if i != j:
            heuristic[i, j] = base[i, j]
            heuristic[j, i] = base[j, i]
    
    # Also add k-NN boost for robustness: edges that are among the 5 nearest
    # neighbors get an additional boost even if not in Delaunay
    k = min(5, n - 1)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        if k < n - 1:
            k_nn = np.argpartition(dists, k)[:k]
        else:
            k_nn = np.where(dists < np.inf)[0]
        for j in k_nn:
            if j != i:
                # Boost k-NN edges that are not already Delaunay
                if heuristic[i, j] < base[i, j]:
                    heuristic[i, j] = base[i, j] * 0.5
    
    # Ensure diagonal is 0 and all off-diagonal values are positive
    np.fill_diagonal(heuristic, 0.0)
    off_diag = ~np.eye(n, dtype=bool)
    heuristic[off_diag] = np.maximum(heuristic[off_diag], 1e-9)
    
    # Normalize
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    # Final safety: ensure positivity
    np.fill_diagonal(heuristic, 0.0)
    heuristic[off_diag] = np.maximum(heuristic[off_diag], 1e-9)
    
    return heuristic


def _reconstruct_points(distance_matrix):
    """Reconstruct 2D point coordinates from a distance matrix using CMDS."""
    n = distance_matrix.shape[0]
    if n < 3:
        return None
    
    D2 = distance_matrix ** 2
    
    # Double centering
    one_n = np.ones((n, n)) / n
    B = -0.5 * (one_n @ D2 @ one_n - one_n @ D2 - D2 @ one_n + D2 @ one_n)
    
    # Eigen decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top 2 dimensions
        if eigenvalues[0] < 1e-10:
            return None
        
        # Scale eigenvectors by sqrt of eigenvalues
        lam = np.sqrt(np.maximum(eigenvalues[:2], 0))
        points = eigenvectors[:, :2] * lam
        
        return points
    except Exception:
        return None


def _delaunay_edge_set(points, n):
    """Compute Delaunay triangulation and return set of edges."""
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    if a < b:
                        edges.add((a, b))
                    else:
                        edges.add((b, a))
        return edges
    except Exception:
        return None


def _knn_heuristic(distance_matrix, base, eps):
    """Fallback heuristic using k-NN boost."""
    n = distance_matrix.shape[0]
    k = max(3, min(10, n // 5))
    
    knn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        if k < n - 1:
            k_nn = np.argpartition(dists, k)[:k]
        else:
            k_nn = np.where(dists < np.inf)[0]
        for j in k_nn:
            knn_mask[i, j] = True
    
    heuristic = np.zeros((n, n), dtype=float)
    knn_edges = knn_mask & (distance_matrix > 0)
    heuristic[knn_edges] = base[knn_edges] ** 1.5
    non_knn = (~knn_mask) & (distance_matrix > 0)
    heuristic[non_knn] = base[non_knn] ** 0.3
    
    np.fill_diagonal(heuristic, 0.0)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    np.fill_diagonal(heuristic, 0.0)
    
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    off_diag = ~np.eye(n, dtype=bool)
    heuristic[off_diag] = np.maximum(heuristic[off_diag], 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
