"""Heuristic matrix for ACO-TSP: MDS-based geometric crossing penalty and MST skeleton."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import distance as sp_dist
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0

def _compute_mds_coordinates(dist_matrix):
    """
    Compute 2D coordinates from distance matrix using Classical MDS.
    Returns None if it fails or takes too long.
    """
    n = dist_matrix.shape[0]
    if n < 4:
        return None
    
    # Check deadline
    if time.time() > _DEADLINE - 30.0:
        return None

    try:
        # Classical MDS
        # Step 1: Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (dist_matrix ** 2) @ J
        
        # Step 2: Eigen decomposition
        # Only need top 2 eigenvalues/vectors
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Sort by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 2
        # If eigenvalues are negative, MDS is not metric, but we can still use the vectors
        # We scale to unit square roughly
        coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
        
        # Center and scale
        coords -= coords.mean(axis=0)
        # Scale to [0, 1] box roughly
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        ranges = max_coords - min_coords
        ranges[ranges == 0] = 1.0
        coords = (coords - min_coords) / ranges
        
        return coords
    except Exception:
        return None

def _check_crossings(coords, edge1, edge2):
    """
    Check if two line segments (edge1, edge2) cross.
    edge1: (p1, p2), edge2: (p3, p4)
    """
    def ccw(A, B, C):
        # Returns True if A, B, C are in counterclockwise order
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    A = coords[edge1[0]]
    B = coords[edge1[1]]
    C = coords[edge2[0]]
    D = coords[edge2[1]]
    
    # Check if endpoints are shared
    if edge1[0] == edge2[0] or edge1[0] == edge2[1] or edge1[1] == edge2[0] or edge1[1] == edge2[1]:
        return False
        
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix.
    
    Strategy:
    1. Compute MDS coordinates from the distance matrix.
    2. Compute MST edges.
    3. Identify edges that cross other "short" edges (proxy for crossing in optimal tour).
    4. Heuristic = (1/d^2) * (1 + MST_Boost) * (Crossing_Penalty)
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Safety check
    if n < 2:
        return np.ones((n, n))
    
    # Base heuristic: Inverse distance squared
    # Add small epsilon to avoid division by zero
    base_heuristic = 1.0 / (d + 1e-9) ** 2
    np.fill_diagonal(base_heuristic, 0.0)
    
    # 1. Compute MST
    # Kruskal's algorithm
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((d[i, j], i, j))
    edges.sort(key=lambda x: x[0])
    
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
    
    mst_edges = set()
    mst_edge_count = 0
    for dist, u, v in edges:
        if union(u, v):
            mst_edges.add((u, v))
            mst_edges.add((v, u))
            mst_edge_count += 1
            if mst_edge_count == n - 1:
                break
    
    # 2. Compute MDS coordinates for crossing detection
    coords = None
    if time.time() < _DEADLINE - 45.0:
        coords = _compute_mds_coordinates(d)
    
    crossing_penalty = np.ones((n, n))
    
    if coords is not None:
        # Identify candidate edges for crossing check
        # We only need to check edges that are "short" enough to be in a tour
        # Sort all edges by distance
        all_edges_list = []
        for i in range(n):
            for j in range(i + 1, n):
                all_edges_list.append((d[i, j], i, j))
        all_edges_list.sort(key=lambda x: x[0])
        
        # Consider top K edges for crossing detection (e.g., 5*n)
        # Edges that are very long are unlikely to be in the tour anyway
        K = min(5 * n, len(all_edges_list))
        candidate_edges = all_edges_list[:K]
        
        # For each candidate edge, check if it crosses any other candidate edge
        # This is O(K^2) which is manageable for n=50 (K=250, 250^2 = 62500 checks)
        # But we can optimize: only check pairs where both are short.
        
        # Initialize penalty to 1.0
        # If an edge crosses another "short" edge, it's likely bad.
        # We will accumulate a penalty score.
        
        # To speed up, we can just check against a subset of the shortest edges
        # The shortest edges are the most likely to be in the tour.
        # Let's check each candidate edge against the top 10 shortest edges.
        top_short_edges = all_edges_list[:10]
        
        for dist, u, v in candidate_edges:
            crosses = False
            for _, su, sv in top_short_edges:
                if (u, v) == (su, sv) or (u, v) == (sv, su):
                    continue
                # Skip if they share a node
                if u == su or u == sv or v == su or v == sv:
                    continue
                if _check_crossings(coords, (u, v), (su, sv)):
                    crosses = True
                    break
            
            if crosses:
                # Apply penalty
                # The penalty should be strong enough to discourage selection
                crossing_penalty[u, v] = 0.1
                crossing_penalty[v, u] = 0.1
    
    # 3. Combine
    # Boost MST edges
    # MST edges are almost always in the TSP tour (or very close).
    # We give them a high attractiveness.
    mst_boost = 5.0
    
    # Non-MST edges get the base heuristic
    # Crossing edges get penalized
    
    final_heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            h = base_heuristic[i, j]
            
            # Apply MST boost
            if (i, j) in mst_edges:
                h *= mst_boost
            
            # Apply crossing penalty
            h *= crossing_penalty[i, j]
            
            final_heuristic[i, j] = h
    
    # Normalize to avoid extreme values
    # ACO uses pheromone * heuristic. If heuristic is too large, it dominates.
    # Let's scale it so the mean is around 1.0 or similar.
    mask = final_heuristic > 0
    if np.any(mask):
        mean_h = np.mean(final_heuristic[mask])
        if mean_h > 0:
            final_heuristic = final_heuristic / mean_h
    
    # Ensure finite and positive
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    np.fill_diagonal(final_heuristic, 0.0)
    
    return final_heuristic
# EVOLVE-BLOCK-END
