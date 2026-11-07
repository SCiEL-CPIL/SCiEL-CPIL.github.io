"""Heuristic matrix for ant colony optimization of the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def _get_points_from_distance_matrix(distance_matrix):
    """Recover 2D point coordinates from a distance matrix using classical MDS."""
    n = distance_matrix.shape[0]
    # Classical MDS: B = -0.5 * J * D^2 * J where J = I - (1/n) * 11^T
    D2 = distance_matrix ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Take top 2 eigenvalues (sorted in descending order)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Coordinates: X = V * sqrt(lambda)
    # Use top 2 dimensions
    if eigenvalues[0] < 0:
        eigenvalues[0] = 0
    if eigenvalues[1] < 0:
        eigenvalues[1] = 0
    X = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    return X


def _compute_delaunay_edges(points):
    """Compute Delaunay triangulation edges as a set of (i, j) pairs."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return edges
    except Exception:
        return set()


def _compute_mst_edges(points):
    """Compute MST edges using Prim's algorithm (simple version for small n)."""
    n = len(points)
    if n == 0:
        return set()
    
    # Compute distance matrix
    D = cdist(points, points)
    in_mst = [False] * n
    min_dist = [np.inf] * n
    min_dist[0] = 0
    edges = set()
    
    for _ in range(n):
        # Find the vertex with minimum distance
        u = -1
        best = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        
        if u == -1:
            break
        
        in_mst[u] = True
        
        for v in range(n):
            if not in_mst[v] and D[u, v] < min_dist[v]:
                min_dist[v] = D[u, v]
                # We'll track parent separately
                if v > u:
                    edges.add((u, v))
                else:
                    edges.add((v, u))
    
    return edges


def _compute_convex_hull(points):
    """Compute convex hull vertices using a simple gift wrapping algorithm."""
    n = len(points)
    if n <= 1:
        return set(range(n))
    
    # Find leftmost point
    start = np.argmin(points[:, 0])
    hull = set()
    hull.add(start)
    
    current = start
    while True:
        # Find the next point on the hull
        next_point = -1
        for i in range(n):
            if i == current or i in hull:
                continue
            if next_point == -1:
                next_point = i
                continue
            # Check if i is to the left of (current, next_point)
            v1 = points[current]
            v2 = points[next_point]
            v3 = points[i]
            cross = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
            if cross < 0:
                next_point = i
            elif cross == 0:
                # Collinear, pick the farther one
                d_next = np.linalg.norm(points[next_point] - points[current])
                d_i = np.linalg.norm(points[i] - points[current])
                if d_i > d_next:
                    next_point = i
        
        if next_point == -1 or next_point == start:
            break
        
        hull.add(next_point)
        current = next_point
    
    return hull


def _compute_local_tangent_alignment(points, edge_set):
    """
    For each potential edge (i, j), compute how well it aligns with local structure.
    Reward edges that are part of Delaunay triangulation or MST, and penalize
    edges that cross other Delaunay/MST edges.
    """
    n = len(points)
    # Start with base score based on distance
    D = cdist(points, points)
    
    # Delaunay edges give a boost
    delaunay_edges = _compute_delaunay_edges(points)
    mst_edges = _compute_mst_edges(points)
    hull_vertices = _compute_convex_hull(points)
    
    # Score matrix: higher is more attractive
    scores = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            dist = D[i, j]
            
            # Base: inverse distance (stronger for closer points)
            base = 1.0 / (dist + 1e-9)
            
            # Boost for Delaunay edges
            boost = 1.0
            if (i, j) in delaunay_edges or (j, i) in delaunay_edges:
                boost *= 2.0
            if (i, j) in mst_edges or (j, i) in mst_edges:
                boost *= 1.5
            if i in hull_vertices and j in hull_vertices:
                # Both on convex hull - likely part of optimal tour
                boost *= 1.3
            
            scores[i, j] = base * boost
            scores[j, i] = base * boost
    
    return scores


def heuristics(distance_matrix):
    """
    Compute a heuristic matrix for ant colony optimization.
    
    Uses classical MDS to recover 2D coordinates, then constructs heuristics
    based on Delaunay triangulation, MST, and convex hull structure.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # Try to recover 2D coordinates
    try:
        points = _get_points_from_distance_matrix(distance_matrix)
    except Exception:
        # Fallback: just use inverse distance
        return 1.0 / (distance_matrix + 1e-9)
    
    # Check time
    if time.time() > deadline:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Compute geometric heuristics
    try:
        scores = _compute_local_tangent_alignment(points, set())
    except Exception:
        # Fallback: just use inverse distance
        return 1.0 / (distance_matrix + 1e-9)
    
    # Check time
    if time.time() > deadline:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Ensure positive values
    scores = np.where(scores > 0, scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
