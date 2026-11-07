import numpy as np
import time
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    """
    Computes a heuristic matrix for ACO TSP using MDS and Delaunay Triangulation.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    
    # Handle invalid values
    if not np.all(np.isfinite(D)):
        D = np.nan_to_num(D, nan=1e9, posinf=1e9, neginf=1e9)
        
    # 1. Classical MDS to get 2D coordinates
    coords = None
    try:
        # Double centering
        # J = I - (1/n) * ones(n, n)
        J = np.eye(n) - np.ones((n, n)) / n
        D_sq = D ** 2
        B = -0.5 * J @ D_sq @ J
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top 2
        if eigenvalues[0] > 0:
            # Ensure non-negative eigenvalues for sqrt
            eigenvalues = np.maximum(eigenvalues, 0)
            coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    except Exception:
        coords = None
        
    # 2. Construct sparse graph edges
    edges = set()
    
    # Add k-NN edges (k=5) for robustness
    # For each node, find 5 nearest neighbors
    # Using argsort on rows of D
    # D[i] is distance from i to all j
    # We want smallest 5
    # Note: D[i,i] is 0, so index i will be first. We skip it.
    try:
        for i in range(n):
            # Get indices of 6 nearest neighbors (including self)
            # argsort is stable? We just take first 6, then skip i
            nearest = np.argsort(D[i])[:6]
            for j_idx in nearest:
                j = int(j_idx)
                if i != j:
                    edges.add((i, j))
                    edges.add((j, i))
    except Exception:
        pass
        
    # Add Delaunay Triangulation edges if coords available
    if coords is not None:
        try:
            # Check for duplicate points or degenerate cases
            # If points are too close, Delaunay might fail or be unstable
            # Simple check: min distance > 0
            min_dist = np.min(D[D > 0]) if np.any(D > 0) else 1.0
            if min_dist > 1e-6:
                tri = Delaunay(coords)
                for simplex in tri.simplices:
                    a, b, c = simplex
                    edges.add((a, b))
                    edges.add((b, a))
                    edges.add((b, c))
                    edges.add((c, b))
                    edges.add((c, a))
                    edges.add((a, c))
        except Exception:
            pass
            
    # 3. Build Heuristic Matrix
    H = np.zeros((n, n), dtype=float)
    
    # For edges in the sparse graph, use inverse distance
    # For others, leave as 0 (evaluator will convert to 1e-9)
    for i, j in edges:
        d = D[i, j]
        if d > 1e-9:
            val = 1.0 / d
        else:
            val = 1.0 / 1e-9
        H[i, j] = val
        H[j, i] = val
        
    # Ensure symmetry and positive values
    # The evaluator handles non-positive -> 1e-9, so zeros are fine.
    
    # Check deadline
    if time.time() > deadline:
        # Fallback to simple inverse distance if we ran out of time (unlikely)
        H = 1.0 / (D + 1e-9)
        
    return H
