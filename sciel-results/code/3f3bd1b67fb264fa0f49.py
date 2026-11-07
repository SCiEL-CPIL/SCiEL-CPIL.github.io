import numpy as np
from scipy.spatial import Delaunay, cKDTree
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit
    
    n = len(coordinates)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Base inverse distance
    d_safe = np.where(dm > 0, dm, 1e-9)
    base = 1.0 / (d_safe + 1e-9)
    np.fill_diagonal(base, 1e-9)
    
    # Build Delaunay triangulation and identify edges
    try:
        tri = Delaunay(coords)
        delaunay_edges = set()
        for simplex in tri.simplices:
            i, j, k = simplex
            delaunay_edges.add((i, j))
            delaunay_edges.add((j, i))
            delaunay_edges.add((j, k))
            delaunay_edges.add((k, j))
            delaunay_edges.add((k, i))
            delaunay_edges.add((i, k))
    except Exception:
        delaunay_edges = set()
    
    # Compute local density using k-nearest neighbors
    try:
        tree = cKDTree(coords)
        # For each node, find its 5 nearest neighbors (excluding self)
        dists, indices = tree.query(coords, k=6)  # k=6 includes self
        # indices[:, 0] is self, indices[:, 1:] are 5 nearest neighbors
        
        # Compute local density: average demand of 5 nearest neighbors
        local_density = np.zeros(n)
        for i in range(n):
            neighbors = indices[i, 1:]  # 5 nearest neighbors
            local_density[i] = np.mean(dem[neighbors])
    except Exception:
        local_density = np.mean(dem[1:]) * np.ones(n)  # Fallback: mean customer demand
    
    # Capacity-aware penalty: penalize edges where combined demand exceeds capacity
    # or where local density is high relative to capacity
    capacity_ratio = capacity / max(np.mean(dem[1:]), 1.0)
    
    heuristic = base.copy()
    
    # Apply capacity-aware penalties
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Non-Delaunay edges get stronger penalty
            if (i, j) not in delaunay_edges:
                heuristic[i, j] *= 0.01
            
            # Capacity-aware penalty: if both nodes have high local density,
            # the edge is less attractive (likely in a dense region where routing is complex)
            density_penalty = 1.0 / (1.0 + 0.1 * (local_density[i] + local_density[j]) / capacity_ratio)
            heuristic[i, j] *= density_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check time budget
    if time.time() > deadline:
        # Return the current heuristic if we're running out of time
        pass
    
    return heuristic
