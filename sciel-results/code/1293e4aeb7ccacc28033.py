"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _covering_radius(points):
    """Compute the covering radius in radians via convex hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2.0  # fallback: 90 degrees in radians
    
    max_angle = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Normal to the plane of the triangle
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        # Ensure normal points outward
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        # Angular distance from circumcenter to each vertex
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        
        facet_radius = max(d1, d2, d3)
        max_angle = max(max_angle, facet_radius)
    
    return max_angle

def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_step(points, hull, lr=0.01):
    """One gradient descent step: move each vertex toward its facet's circumcenter
    to reduce the maximum circumradius."""
    n = len(points)
    # Accumulate gradient for each point
    grad = np.zeros_like(points)
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        # The circumcenter direction is `normal`
        # Gradient: each vertex should move toward the circumcenter direction
        # The angular distance from vertex vi to circumcenter is arccos(dot(normal, vi))
        # To reduce the max, move vi in the direction of (normal - vi) projected tangent
        
        for idx in simplex:
            v = points[idx]
            # Tangent component of (normal - v)
            radial = np.dot(normal, v) * v
            tangent_dir = normal - radial
            tnorm = np.linalg.norm(tangent_dir)
            if tnorm > 1e-10:
                grad[idx] += tangent_dir / tnorm
    
    # Normalize gradient per point
    for i in range(n):
        gnorm = np.linalg.norm(grad[i])
        if gnorm > 1e-10:
            grad[i] /= gnorm
    
    # Apply gradient descent
    points = points - lr * grad
    
    # Re-normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points

def _lloyd_iteration(points, iterations=50, deadline=None):
    """Lloyd's algorithm: move each point to the centroid of its Voronoi cell (on the sphere)."""
    points = points.copy().astype(float)
    n = len(points)
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # For each point, find its Voronoi cell on the sphere
        # Approximate: for each point, the cell is the set of points closer to it than any other
        # We can approximate the centroid by averaging all points on the sphere that are
        # in its Voronoi cell. A simpler approach: use the convex hull facets.
        
        # Alternative: use a grid-based approach to find the centroid of each Voronoi cell
        # Create a fine grid on the sphere
        n_grid = 50
        theta = np.linspace(0, np.pi, n_grid)
        phi = np.linspace(0, 2 * np.pi, 2 * n_grid, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        grid_points = np.column_stack([
            np.sin(TH) * np.cos(PH).flatten(),
            np.sin(TH) * np.sin(PH).flatten(),
            np.cos(TH).flatten()
        ])
        
        # For each grid point, find the nearest configuration point
        # dots[grid_idx, point_idx] = dot(grid_point, config_point)
        dots = grid_points @ points.T
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroid for each configuration point
        new_points = np.zeros_like(points)
        counts = np.zeros(n)
        for i in range(n):
            mask = nearest == i
            if np.sum(mask) > 0:
                new_points[i] = grid_points[mask].sum(axis=0)
                counts[i] = np.sum(mask)
        
        # Normalize new points
        for i in range(n):
            if counts[i] > 0:
                new_points[i] /= np.linalg.norm(new_points[i])
            else:
                new_points[i] = points[i]
        
        # Blend with old points for stability
        alpha = 0.5
        points = (1 - alpha) * points + alpha * new_points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return points

def _gradient_optimize(points, max_iters=200, lr=0.02, deadline=None):
    """Gradient descent optimization using facet circumcenter gradients."""
    points = points.copy().astype(float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Adaptive learning rate
        current_lr = lr * (0.95 ** (it // 20))
        
        new_points = _gradient_step(points, hull, lr=current_lr)
        
        # Check if improvement
        old_radius = _covering_radius(points)
        new_radius = _covering_radius(new_points)
        
        if new_radius < old_radius:
            points = new_points
        else:
            # Try smaller step
            new_points = _gradient_step(points, hull, lr=current_lr * 0.1)
            new_radius = _covering_radius(new_points)
            if new_radius < old_radius:
                points = new_points
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Start 1: Fibonacci spiral
    points = _fibonacci_sphere(N)
    radius = _covering_radius(points)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Optimize with gradient descent
    if time.time() < deadline - 30:
        opt_points = _gradient_optimize(points, max_iters=100, lr=0.02, deadline=deadline - 25)
        opt_radius = _covering_radius(opt_points)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_points = opt_points.copy()
    
    # Start 2: Different Fibonacci offset
    if time.time() < deadline - 30:
        points2 = _fibonacci_sphere(N, offset=0.1)
        radius2 = _covering_radius(points2)
        if radius2 < best_radius:
            best_radius = radius2
            best_points = points2.copy()
        
        if time.time() < deadline - 25:
            opt_points2 = _gradient_optimize(points2, max_iters=100, lr=0.02, deadline=deadline - 20)
            opt_radius2 = _covering_radius(opt_points2)
            if opt_radius2 < best_radius:
                best_radius = opt_radius2
                best_points = opt_points2.copy()
    
    # Start 3: Random initialization
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            rand_points = rng.normal(size=(N, 3))
            rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
            
            # Optimize random start
            if time.time() < deadline - 15:
                opt_rand = _gradient_optimize(rand_points, max_iters=150, lr=0.03, deadline=deadline - 10)
                opt_rand_radius = _covering_radius(opt_rand)
                if opt_rand_radius < best_radius:
                    best_radius = opt_rand_radius
                    best_points = opt_rand.copy()
    
    # Final refinement
    if best_points is not None and time.time() < deadline - 5:
        final = _gradient_optimize(best_points, max_iters=100, lr=0.005, deadline=deadline - 2)
        final_radius = _covering_radius(final)
        if final_radius < best_radius:
            best_points = final
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
