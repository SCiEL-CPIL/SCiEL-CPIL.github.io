"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _covering_radius(points):
    """
    Compute the exact covering radius (in radians) using the convex hull.
    The covering radius is the maximum circumradius of the spherical facets.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex]
            normal = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            d = np.dot(normal, v1)
            if d < 0:
                normal = -normal
            dot = np.dot(normal, v1)
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            if angle > max_r:
                max_r = angle
        return max_r
    except Exception:
        return np.pi

def _lloyd_step(points):
    """
    One step of Lloyd's algorithm on the sphere.
    For each point, compute its Voronoi cell (spherical polygon) and move
    the point to the centroid of that cell.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    # For each point, find all facets (triangles) that contain it
    # Build a mapping from point index to list of facet normals
    point_facets = [[] for _ in range(len(points))]
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex]
        normal = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-12:
            continue
        normal = normal / norm_n
        d = np.dot(normal, v1)
        if d < 0:
            normal = -normal
        # This facet is associated with all three vertices
        for idx in simplex:
            point_facets[idx].append(normal)
    
    new_points = points.copy()
    for i in range(len(points)):
        if not point_facets[i]:
            continue
        # The centroid of the Voronoi cell is the normalized sum of facet normals
        # weighted by the solid angle subtended by each facet
        # For simplicity, use uniform weighting first
        centroid = np.sum(point_facets[i], axis=0)
        norm_c = np.linalg.norm(centroid)
        if norm_c < 1e-12:
            continue
        centroid = centroid / norm_c
        new_points[i] = centroid
    
    return new_points

def _lloyd_optimize(points, max_iters=500, tol=1e-10):
    """
    Run Lloyd's algorithm to minimize the covering radius.
    """
    current = points.copy()
    best = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        new_points = _lloyd_step(current)
        
        # Check for convergence
        if np.allclose(current, new_points, atol=tol):
            break
        
        cr = _covering_radius(new_points)
        if cr < best_cr:
            best_cr = cr
            best = new_points.copy()
        
        current = new_points
    
    return best, best_cr

def _repulsive_force_step(points, k=100, steps=50, lr=0.01):
    """
    A repulsive force-based optimization step.
    Points repel each other with a force proportional to 1/d^2.
    """
    current = points.copy()
    for step in range(steps):
        forces = np.zeros_like(current)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = current[i] - current[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    continue
                # Repulsive force: k / dist^2 in direction away from j
                force = k * d / (dist ** 3)
                forces[i] += force
        
        # Add a small random perturbation to escape local minima
        noise = np.random.randn(N, 3) * 0.001
        forces += noise
        
        # Update positions
        current += lr * forces
        # Normalize back to unit sphere
        norms = np.linalg.norm(current, axis=1)
        current = current / norms[:, np.newaxis]
    
    return current

def _simulated_annealing(points, start_temp=0.1, end_temp=0.001, max_iters=200):
    """
    Simulated annealing to escape local minima.
    """
    current = points.copy()
    best = points.copy()
    best_cr = _covering_radius(points)
    
    n_iters = max_iters
    for it in range(n_iters):
        temp = start_temp * (end_temp / start_temp) ** (it / max(1, n_iters - 1))
        
        # Generate a random perturbation
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rotate the point by a small angle around a random axis
        theta = np.random.uniform(-temp, temp)
        # Rodrigues rotation formula
        c = np.cos(theta)
        s = np.sin(theta)
        rotated = (current[idx] * c + np.cross(axis, current[idx]) * s + 
                   axis * np.dot(axis, current[idx]) * (1 - c))
        new_points = current.copy()
        new_points[idx] = rotated / np.linalg.norm(rotated)
        
        cr_new = _covering_radius(new_points)
        cr_cur = _covering_radius(current)
        
        # Accept if better or with probability exp(-delta/temp)
        if cr_new < cr_cur or np.random.rand() < np.exp(-(cr_new - cr_cur) / temp):
            current = new_points
            if cr_new < best_cr:
                best_cr = cr_new
                best = new_points.copy()
    
    return best, best_cr

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin (evaluator kills at 150s)
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Run Lloyd's algorithm
    best_points, best_cr = _lloyd_optimize(points, max_iters=300)
    
    # If we have time left, try multiple random restarts
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        np.random.seed(42)
        for restart in range(8):
            if time.time() - start_time > time_limit - 10.0:
                break
            # Random perturbation of the initial points
            perturbed = points.copy()
            noise = np.random.randn(N, 3) * 0.1
            perturbed = perturbed + noise
            norms = np.linalg.norm(perturbed, axis=1)
            perturbed = perturbed / norms[:, np.newaxis]
            
            restarted_points, restarted_cr = _lloyd_optimize(perturbed, max_iters=100)
            if restarted_cr < best_cr:
                best_cr = restarted_cr
                best_points = restarted_points.copy()
    
    # If we still have time, try simulated annealing from the best point
    remaining = time_limit - (time.time() - start_time)
    if remaining > 20.0:
        sa_points, sa_cr = _simulated_annealing(best_points, start_temp=0.05, end_temp=0.001, max_iters=100)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points.copy()
    
    # Final polish with more Lloyd iterations if time permits
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        polished_points, polished_cr = _lloyd_optimize(best_points, max_iters=200, tol=1e-12)
        if polished_cr < best_cr:
            best_cr = polished_cr
            best_points = polished_points.copy()
    
    return best_points

# EVOLVE-BLOCK-END
