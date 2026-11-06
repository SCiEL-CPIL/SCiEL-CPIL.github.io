"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
    dists = np.sqrt(dists_sq)
    log_dists = np.log(dists)
    np.fill_diagonal(log_dists, 0.0)
    energy = -0.5 * np.sum(log_dists)
    return energy

def _lloyd_step(points, iterations=50):
    """
    Perform Lloyd's algorithm steps to minimize logarithmic energy.
    For each point, compute its Voronoi cell on the sphere and move it to the centroid.
    """
    n = len(points)
    pts = points.copy()
    
    for _ in range(iterations):
        # Compute Voronoi cells using convex hull of points on the sphere
        # We use a trick: project to 3D, compute convex hull, and use face normals
        # Actually, a more robust way for sphere Voronoi is to use the dual of the convex hull
        # But scipy doesn't have direct spherical Voronoi. 
        # Alternative: Use the fact that for points on a sphere, the Voronoi cell of p_i
        # is the set of points x such that |x - p_i| <= |x - p_j| for all j.
        # On the sphere, this is equivalent to p_i . x >= p_j . x for all j.
        # So the cell is the intersection of hemispheres defined by p_i - p_j.
        
        # A simpler approximation for Lloyd on sphere:
        # Move each point towards the centroid of its neighbors within a certain distance.
        # Or, use the gradient of the energy, which is known to be related to the Voronoi cell.
        
        # Actually, the gradient of the logarithmic energy on the sphere is:
        # grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # This is exactly the vector sum of repulsive forces.
        # Lloyd's method is equivalent to gradient descent with a specific step size.
        
        # Let's use a simple gradient descent step with adaptive step size.
        # This is more robust than trying to compute Voronoi cells directly.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        grad = -np.sum(diff / dists_sq[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        grad_tangent = np.zeros_like(grad)
        for i in range(n):
            proj = np.dot(grad[i], pts[i])
            grad_tangent[i] = grad[i] - proj * pts[i]
        
        # Adaptive step size
        # Use a small step size to ensure stability
        step_size = 0.01
        new_pts = pts + step_size * grad_tangent
        new_pts = _normalize(new_pts)
        
        # Check if energy decreased
        old_energy = _log_energy(pts)
        new_energy = _log_energy(new_pts)
        
        if new_energy < old_energy:
            pts = new_pts
        else:
            # If energy increased, reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
            new_pts = pts + step_size * grad_tangent
            new_pts = _normalize(new_pts)
            pts = new_pts
    
    return pts

def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _gradient_polish(points, deadline, maxiter=100):
    """Polish the points using L-BFGS-B with analytic gradient."""
    n = len(points)
    
    def objective_and_grad(x):
        pts = _normalize(x.reshape(-1, 3))
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        log_dists = np.log(dists)
        np.fill_diagonal(log_dists, 0.0)
        energy = -0.5 * np.sum(log_dists)
        
        grad = -np.sum(diff / dists_sq[:, :, np.newaxis], axis=1)
        grad_tangent = np.zeros_like(grad)
        for i in range(n):
            proj = np.dot(grad[i], pts[i])
            grad_tangent[i] = grad[i] - proj * pts[i]
        
        return energy, grad_tangent.flatten()
    
    result = minimize(
        objective_and_grad,
        points.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'maxls': 50}
    )
    
    return _normalize(result.x.reshape(-1, 3))

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)
    
    # Evaluate initial Fibonacci
    try:
        e = _log_energy(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    rng = np.random.default_rng(42)
    
    # 1. Lloyd's algorithm from Fibonacci start
    try:
        if time.time() < deadline - 30:
            lloyd_pts = _lloyd_step(best_points, iterations=100)
            e = _log_energy(lloyd_pts)
            if e < best_energy:
                best_energy = e
                best_points = lloyd_pts
    except Exception:
        pass

    # 2. Gradient polish
    try:
        if time.time() < deadline - 10:
            polished = _gradient_polish(best_points, deadline, maxiter=100)
            e = _log_energy(polished)
            if e < best_energy:
                best_energy = e
                best_points = polished
    except Exception:
        pass

    # 3. Multi-start with Lloyd's algorithm
    num_restarts = 20
    for k in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random start
        pts = _random_sphere(N, rng)
        
        try:
            if time.time() < deadline - 15:
                # Lloyd's algorithm
                lloyd_pts = _lloyd_step(pts, iterations=50)
                e = _log_energy(lloyd_pts)
                
                if e < best_energy:
                    best_energy = e
                    best_points = lloyd_pts
                    
                    # Gradient polish if good
                    if time.time() < deadline - 10:
                        polished = _gradient_polish(lloyd_pts, deadline, maxiter=50)
                        e2 = _log_energy(polished)
                        if e2 < best_energy:
                            best_energy = e2
                            best_points = polished
        except Exception:
            continue

    # 4. Final polish
    try:
        if time.time() < deadline - 5:
            final_pts = _gradient_polish(best_points, deadline, maxiter=200)
            e = _log_energy(final_pts)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
    except Exception:
        pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
