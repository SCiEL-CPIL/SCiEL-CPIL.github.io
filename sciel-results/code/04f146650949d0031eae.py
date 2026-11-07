"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = dm.shape[0]

    # Work on a copy to avoid mutating input
    d = dm.copy()

    # Set diagonal to a large value so it gets low attractiveness
    np.fill_diagonal(d, np.inf)

    # Ensure no negative distances (shouldn't happen, but safe)
    d = np.maximum(d, 1e-9)

    # --- Step 1: Compute 2D embedding via classical MDS ---
    # For n=50 points in unit square, MDS should recover the geometry well
    # since the distance matrix is Euclidean.
    
    # Classical MDS: 
    # 1. Double centering: B = -1/2 * J * D^2 * J where J = I - (1/n)11^T
    # 2. Eigenvalue decomposition of B
    # 3. Take top 2 eigenvectors scaled by sqrt of eigenvalues
    
    D2 = d ** 2
    # Set diagonal to 0 for MDS
    np.fill_diagonal(D2, 0.0)
    
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 eigenvalues (should be positive for Euclidean distances)
    # Clamp negative eigenvalues to 0
    top2_evals = np.maximum(eigenvalues[:2], 0.0)
    top2_evecs = eigenvectors[:, :2]
    
    # Embedding: X = V * sqrt(lambda)
    X = top2_evecs * np.sqrt(top2_evals)
    
    # Check time
    if time.time() > deadline - 10.0:
        # Fallback to simple inverse distance if MDS took too long
        h = 1.0 / np.power(d, 2.0)
        mask = ~np.eye(n, dtype=bool)
        mean_h = np.mean(h[mask])
        if mean_h > 0 and np.isfinite(mean_h):
            h = h / mean_h
        else:
            h = np.ones_like(dm, dtype=float)
        h = np.where(np.isfinite(h), h, 1e-9)
        h = np.where(h > 0, h, 1e-9)
        return h

    # --- Step 2: Compute Delaunay triangulation on the MDS embedding ---
    try:
        tri = Delaunay(X)
        delaunay_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a != b:
                        delaunay_edges.add((min(a, b), max(a, b)))
    except Exception:
        # Fallback if Delaunay fails
        delaunay_edges = set()
    
    # Check time
    if time.time() > deadline - 10.0:
        h = 1.0 / np.power(d, 2.0)
        mask = ~np.eye(n, dtype=bool)
        mean_h = np.mean(h[mask])
        if mean_h > 0 and np.isfinite(mean_h):
            h = h / mean_h
        else:
            h = np.ones_like(dm, dtype=float)
        h = np.where(np.isfinite(h), h, 1e-9)
        h = np.where(h > 0, h, 1e-9)
        return h

    # --- Step 3: Compute tangent directions for each node from MST ---
    # Build MST using Kruskal's algorithm on the Delaunay edges (or all edges)
    # For n=50, we can use all edges with a threshold
    
    # Use all pairs for MST (n=50, so 1225 edges)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((d[i, j], i, j))
    edges.sort()
    
    # Kruskal's algorithm
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    mst_edges = []
    for dist, i, j in edges:
        if union(i, j):
            mst_edges.append((i, j))
        if len(mst_edges) == n - 1:
            break
    
    # Check time
    if time.time() > deadline - 10.0:
        h = 1.0 / np.power(d, 2.0)
        mask = ~np.eye(n, dtype=bool)
        mean_h = np.mean(h[mask])
        if mean_h > 0 and np.isfinite(mean_h):
            h = h / mean_h
        else:
            h = np.ones_like(dm, dtype=float)
        h = np.where(np.isfinite(h), h, 1e-9)
        h = np.where(h > 0, h, 1e-9)
        return h

    # Compute tangent direction for each node based on MST neighbors
    # For each node, the tangent is the average direction of its MST neighbors
    tangent_dirs = np.zeros((n, 2))
    for i in range(n):
        neighbors = []
        for a, b in mst_edges:
            if a == i:
                neighbors.append(b)
            elif b == i:
                neighbors.append(a)
        
        if len(neighbors) > 0:
            # Average direction to neighbors
            total_dir = np.zeros(2)
            for nb in neighbors:
                diff = X[nb] - X[i]
                norm = np.linalg.norm(diff)
                if norm > 1e-12:
                    total_dir += diff / norm
            norm_total = np.linalg.norm(total_dir)
            if norm_total > 1e-12:
                tangent_dirs[i] = total_dir / norm_total
            else:
                # Random direction if no clear tangent
                angle = np.random.uniform(0, 2 * np.pi)
                tangent_dirs[i] = np.array([np.cos(angle), np.sin(angle)])
        else:
            # Isolated node (shouldn't happen in MST)
            angle = np.random.uniform(0, 2 * np.pi)
            tangent_dirs[i] = np.array([np.cos(angle), np.sin(angle)])

    # --- Step 4: Compute angular deviation heuristic ---
    # For each potential edge (i, j), compute the angle between:
    # 1. The direction from i to j
    # 2. The tangent direction at i
    # Reward edges that are aligned with the local tangent (small angular deviation)
    
    # Also consider Delaunay edges as a boost
    
    h_base = 1.0 / np.power(d, 2.0)
    
    # Compute angular alignment score
    angular_score = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                angular_score[i, j] = 0.0
                continue
            
            # Direction from i to j
            diff = X[j] - X[i]
            norm_diff = np.linalg.norm(diff)
            if norm_diff < 1e-12:
                angular_score[i, j] = 1.0
                continue
            
            dir_ij = diff / norm_diff
            
            # Tangent at i
            tangent = tangent_dirs[i]
            norm_tangent = np.linalg.norm(tangent)
            if norm_tangent < 1e-12:
                angular_score[i, j] = 1.0
                continue
            
            tangent = tangent / norm_tangent
            
            # Cosine of angle between dir_ij and tangent
            cos_angle = np.dot(dir_ij, tangent)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Score: higher when aligned (cos_angle close to 1)
            # Use a smooth function: score = 1 + cos_angle (range [0, 2])
            # Or use exponential: score = exp(cos_angle) for sharper contrast
            angular_score[i, j] = 1.0 + 0.5 * cos_angle  # Range [0.5, 1.5]
    
    # Check time
    if time.time() > deadline - 5.0:
        h = h_base * angular_score
        mask = ~np.eye(n, dtype=bool)
        mean_h = np.mean(h[mask])
        if mean_h > 0 and np.isfinite(mean_h):
            h = h / mean_h
        else:
            h = np.ones_like(dm, dtype=float)
        h = np.where(np.isfinite(h), h, 1e-9)
        h = np.where(h > 0, h, 1e-9)
        return h

    # Combine: base distance heuristic * angular alignment * Delaunay boost
    delaunay_mask = np.zeros((n, n), dtype=bool)
    for a, b in delaunay_edges:
        delaunay_mask[a, b] = True
        delaunay_mask[b, a] = True
    
    boost_factor = 1.5
    h = h_base * angular_score
    h[delaunay_mask] *= boost_factor

    # Normalize to mean 1.0 over off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    valid_h = h[mask]
    if len(valid_h) > 0:
        mean_h = np.mean(valid_h)
        if mean_h > 0 and np.isfinite(mean_h):
            h = h / mean_h
        else:
            h = np.ones_like(dm, dtype=float)
    else:
        h = np.ones_like(dm, dtype=float)

    # Ensure non-finite values are handled
    h = np.where(np.isfinite(h), h, 1e-9)
    
    # Entries that are not positive become 1e-9
    h = np.where(h > 0, h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
