"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_covering_radius(points):
    """Compute the covering radius (in degrees) for points on unit sphere."""
    try:
        hull = ConvexHull(points)
        max_cos = 1.0
        for simplex in hull.simplices:
            verts = points[simplex]
            if len(verts) < 3:
                continue
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            dist = np.abs(np.dot(normal, v1))
            dist = np.clip(dist, -1.0, 1.0)
            max_cos = min(max_cos, dist)
        
        if max_cos > 1.0:
            max_cos = 1.0
        if max_cos < -1.0:
            max_cos = -1.0
        
        radius_rad = np.arccos(max_cos)
        return np.degrees(radius_rad)
    except Exception:
        return 180.0


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points


def _compute_facet_data(points):
    """Compute facet normals, circumradii, and centers for the convex hull."""
    try:
        hull = ConvexHull(points)
        n_facets = len(hull.simplices)
        facet_normals = np.zeros((n_facets, 3))
        facet_circumradii = np.zeros(n_facets)
        facet_centers = np.zeros((n_facets, 3))
        
        for i, simplex in enumerate(hull.simplices):
            verts = points[simplex]
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            facet_normals[i] = normal
            
            # Circumradius in angular space
            dist = np.abs(np.dot(normal, v1))
            dist = np.clip(dist, -1.0, 1.0)
            radius_rad = np.arccos(dist)
            facet_circumradii[i] = np.degrees(radius_rad)
            
            # Facet center (circumcenter on sphere)
            facet_centers[i] = normal
        
        return facet_normals, facet_circumradii, facet_centers, hull
    except Exception:
        return None, None, None, None


def _lloyd_iteration(points, n_iter=20, time_limit=30.0):
    """Lloyd's algorithm: move each point to the centroid of its Voronoi cell."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Voronoi cells using convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (intersection of half-spaces)
        # Approximate: for each point, compute weighted average of nearby points
        # where weights are inverse distance squared
        new_points = np.zeros_like(points)
        for i in range(N):
            # Find k nearest neighbors
            dists = np.linalg.norm(points - points[i], axis=1)
            k = min(10, N)
            nn_idx = np.argsort(dists)[:k]
            
            # Weighted average with inverse distance weights
            weights = 1.0 / (dists[nn_idx] + 1e-10)
            weights[i] = 0  # Don't include self
            weights /= weights.sum() + 1e-10
            
            centroid = np.sum(points[nn_idx] * weights[:, None], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_points[i] = centroid / norm
            else:
                new_points[i] = points[i]
        
        # Check convergence
        if np.max(np.linalg.norm(new_points - points, axis=1)) < 1e-12:
            points = new_points
            break
        
        points = new_points
        
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points, best_radius


def _simulated_annealing(points, time_limit=60.0):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    T_init = 0.1
    T_min = 0.001
    n_steps = 500
    
    current_points = points.copy()
    current_radius = best_radius
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Annealing schedule
        T = T_init * (T_min / T_init) ** (step / n_steps)
        
        # Generate perturbation: random small rotation
        angle = T * 0.1
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
        
        new_points = current_points @ R.T
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        # Accept with probability based on temperature
        if new_radius < current_radius or np.random.random() < np.exp(-(new_radius - current_radius) / max(T, 1e-10)):
            current_points = new_points
            current_radius = new_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
    
    return best_points, best_radius


def _hybrid_optimize(points, time_limit=90.0):
    """Combine Lloyd, simulated annealing, and Lloyd again."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Phase 1: Lloyd
    t1 = min(20.0, (time_limit - (time.time() - start_time)) * 0.3)
    points, radius = _lloyd_iteration(points, n_iter=15, time_limit=t1)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Phase 2: Simulated annealing
    t2 = min(40.0, (time_limit - (time.time() - start_time)) * 0.5)
    points, radius = _simulated_annealing(points, time_limit=t2)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Phase 3: Lloyd again
    t3 = min(20.0, time_limit - (time.time() - start_time))
    if t3 > 5:
        points, radius = _lloyd_iteration(points, n_iter=10, time_limit=t3)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = 180.0
    
    # Multiple restarts with different initializations
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initializations for each restart
        if restart == 0:
            points = _fibonacci_spiral()
        elif restart == 1:
            points = _fibonacci_spiral()
            points += np.random.randn(N, 3) * 0.1
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            points = _fibonacci_spiral()
            points += np.random.randn(N, 3) * 0.05
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            # Random with some structure
            i = np.arange(N)
            z = 1.0 - 2.0 * i / N
            phi = np.random.uniform(0, 2 * np.pi, N)
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            points = _fibonacci_spiral()
            # Rotate
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
            points = points @ rot.T
        
        # Hybrid optimization
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        
        opt_time = min(90.0, remaining_time * 0.8)
        points, radius = _hybrid_optimize(points, time_limit=opt_time)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    return best_points
# EVOLVE-BLOCK-END

