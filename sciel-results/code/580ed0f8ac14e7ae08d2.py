import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    
    # Reconstruct 2D coordinates using Classical MDS
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ (distance_matrix ** 2) @ J)
    B = (B + B.T) / 2
    
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    eigvals = np.maximum(eigvals, 0)
    coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])[:, np.newaxis]
    
    try:
        tri = Delaunay(coords)
        simplices = tri.simplices
    except Exception:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Initialize heuristic matrix
    heuristic = np.zeros_like(distance_matrix)
    
    # Identify Delaunay edges
    delaunay_edges = set()
    for simplex in simplices:
        v0, v1, v2 = simplex
        delaunay_edges.add((min(v0, v1), max(v0, v1)))
        delaunay_edges.add((min(v0, v2), max(v0, v2)))
        delaunay_edges.add((min(v1, v2), max(v1, v2)))
    
    # Fill heuristic matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 0.0
            else:
                d = distance_matrix[i, j]
                if (min(i, j), max(i, j)) in delaunay_edges:
                    heuristic[i, j] = 10.0 / (d + 1e-9)
                else:
                    heuristic[i, j] = 0.1 / (d + 1e-9)
                    
    return heuristic
