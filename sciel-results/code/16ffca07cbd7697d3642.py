"""Rewritten program for sphere covering with 50 points using Gram matrix optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _normalize_points(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _get_circumradius(p1, p2, p3):
    """
    Calculate the circumradius of a spherical triangle defined by 3 unit vectors.
    Returns (radius_in_radians, center_unit_vector).
    """
    n = np.cross(p1 - p2, p2 - p3)
    c = _normalize(n)
    if np.dot(c, p1) < 0:
        c = -c
    cos_r = np.dot(c, p1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    r = np.arccos(cos_r)
    return r, c

def _compute_covering_radius(points):
    """
    Compute the covering radius of the configuration in radians.
    Returns max_r, worst_facet_indices, worst_center.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        worst_facet = None
        worst_center = None
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            r, c = _get_circumradius(p1, p2, p3)
            if r > max_r:
                max_r = r
                worst_facet = simplex
                worst_center = c
        return max_r, worst_facet, worst_center
    except Exception:
        return np.pi, None, None

def _gram_to_points(G):
    """
    Convert a Gram matrix to points on the sphere.
    Uses Cholesky decomposition or eigenvalue decomposition for robustness.
    """
    G = 0.5 * (G + G.T)
    # Add small regularization for numerical stability
    G_reg = G + 1e-10 * np.eye(N)
    try:
        L = np.linalg.cholesky(G_reg)
        points = L.T
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(G_reg)
        eigvals = np.clip(eigvals, 0, None)
        points = eigvecs * np.sqrt(eigvals)
    
    # Normalize columns to unit vectors
    norms = np.linalg.norm(points, axis=0, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    return points.T  # Shape (N, 3)

def _points_to_gram(points):
    """Convert points to Gram matrix."""
    return points @ points.T

def _pairwise_distances_from_gram(G):
    """
    Compute pairwise angular distances from Gram matrix.
    G[i,j] = cos(theta_ij)
    """
    G = 0.5 * (G + G.T)
    np.fill_diagonal(G, 1.0)
    G = np.clip(G, -1.0, 1.0)
    # Angular distances
    D = np.arccos(G)
    return D

def _smooth_min_pairwise_distance(params, beta=50.0):
    """
    Smooth surrogate for the minimum pairwise distance using log-sum-exp.
    We want to MAXIMIZE the minimum pairwise distance.
    So we minimize -min(D_ij) ≈ -logsumexp(-beta * D_ij) / beta
    """
    G = params.reshape(N, N)
    G = 0.5 * (G + G.T)
    np.fill_diagonal(G, 1.0)
    G = np.clip(G, -1.0 + 1e-8, 1.0 - 1e-8)
    
    D = np.arccos(G)
    # Zero out diagonal
    D = np.where(np.eye(N, dtype=bool), 0.0, D)
    
    # We want to maximize min(D_ij) for i != j
    # This is equivalent to minimizing -min(D_ij)
    # Smooth approximation: -logsumexp(-beta * D) / beta
    # But we only want off-diagonal elements
    mask = ~np.eye(N, dtype=bool)
    D_off = D[mask]
    
    if len(D_off) == 0:
        return 0.0
    
    # logsumexp(-beta * D_off) / beta
    # = max(-beta * D_off) / beta + log(sum(exp(-beta * D_off - max(-beta * D_off)))) / beta
    # = -max(D_off) + log(sum(exp(-beta * D_off + max(D_off)))) / beta
    
    max_val = np.max(-beta * D_off)
    logsumexp = max_val + np.log(np.sum(np.exp(-beta * D_off - max_val)))
    return -logsumexp / beta

def _lbfgs_gram_optimize(points, time_limit=30.0, beta=50.0):
    """
    Optimize the Gram matrix to maximize minimum pairwise distance.
    """
    start_time = time.time()
    G0 = _points_to_gram(points)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_min_pairwise_distance(params, beta=beta)
    
    try:
        result = minimize(objective, G0.flatten(), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500})
        G_opt = result.x.reshape(N, N)
        points_opt = _gram_to_points(G_opt)
        return points_opt
    except Exception:
        return points

def _lloyd_relaxation(points, max_iter=100, time_limit=20.0):
    """
    Lloyd's relaxation: iteratively move each point to the centroid of its Voronoi cell.
    """
    start_time = time.time()
    points = points.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # For each point, find its Voronoi cell centroid
        # This is equivalent to finding the point on the sphere that maximizes
        # the minimum distance to all other points, subject to being in the same cell.
        # Approximation: use the centroid of the convex hull of the cell.
        
        new_points = np.zeros_like(points)
        for i in range(N):
            # Find all points that are closer to points[i] than to any other point
            # This is expensive, so we use a simpler heuristic:
            # Move point i towards the centroid of its neighbors
            
            # Compute distances to all other points
            dots = points @ points[i]
            # Points in the same Voronoi cell are those for which
            # dot(p_i, p_j) > dot(p_i, p_k) for all k != i, j
            # This is complex, so we use a simpler approach:
            # Move towards the average of points that are "close" but not too close
            
            # Simple heuristic: move towards the direction that increases
            # the minimum distance to other points
            min_dist = np.inf
            min_idx = -1
            for j in range(N):
                if j == i:
                    continue
                d = np.arccos(np.clip(np.dot(points[i], points[j]), -1, 1))
                if d < min_dist:
                    min_dist = d
                    min_idx = j
            
            if min_idx >= 0:
                # Move away from the closest point
                v = points[i] - points[min_idx]
                v = _normalize(v)
                # Blend with original position
                new_points[i] = _normalize(points[i] + 0.5 * v)
            else:
                new_points[i] = points[i]
        
        points = new_points
    
    return points

def _simulated_annealing(points, start_time, deadline, T0=0.1, T_end=0.001, cooling=0.995):
    """
    Simulated annealing to escape local minima.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    T = T0
    while time.time() < deadline - 5.0 and T > T_end:
        # Propose a random perturbation
        i = np.random.randint(N)
        # Random direction on the sphere
        v = np.random.randn(3)
        v = _normalize(v)
        # Perturb point i
        step = T * 0.5
        new_points = points.copy()
        new_points[i] = _normalize(points[i] + step * v)
        
        new_radius, _, _ = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
        
        # Accept or reject
        delta = new_radius - _compute_covering_radius(points)[0]
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            points = new_points
        
        T *= cooling
    
    return best_points, best_radius

def _local_refine(points, time_limit=10.0):
    """
    Local refinement using facet shrinking.
    """
    start_time = time.time()
    points = points.copy()
    step = 0.05
    
    for it in range(500):
        if time.time() - start_time > time_limit:
            break
        
        current_radius, worst_facet, worst_center = _compute_covering_radius(points)
        
        if worst_facet is None:
            break
        
        # Move the vertices of the worst facet towards the circumcenter
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            grad = c - np.dot(c, v) * v
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return v
            grad = grad / grad_norm
            v_new = v + step * grad
            return _normalize(v_new)
        
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        if it % 50 == 0:
            step *= 0.9
            if step < 0.001:
                step = 0.001
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    # Phase 1: Gram matrix optimization to maximize minimum pairwise distance
    if time.time() < deadline - 60.0:
        points = _lbfgs_gram_optimize(points, time_limit=30.0, beta=30.0)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Second Gram optimization with higher beta
    if time.time() < deadline - 40.0:
        points = _lbfgs_gram_optimize(points, time_limit=20.0, beta=100.0)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 3: Lloyd's relaxation
    if time.time() < deadline - 20.0:
        points = _lloyd_relaxation(points, max_iter=50, time_limit=15.0)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 10.0:
        points, radius = _simulated_annealing(points, start_time, deadline, T0=0.05, T_end=0.005, cooling=0.998)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 5: Local refinement
    if time.time() < deadline - 5.0:
        points = _local_refine(best_points, time_limit=8.0)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Final validation
    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return points
