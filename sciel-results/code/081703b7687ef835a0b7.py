"""Optimized program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time


def _spherical_grid(n_lat=100, n_lon=200):
    """Generate a dense spherical grid for evaluating covering radius."""
    # Use Fibonacci spiral for more uniform coverage
    i = np.arange(n_lat * n_lon) + 0.5
    z = 1.0 - 2.0 * i / (n_lat * n_lon)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius(points, grid=None):
    """Compute the covering radius in radians using a dense grid."""
    if grid is None:
        grid = _spherical_grid()
    
    # Find nearest neighbor for each grid point
    tree = cKDTree(points)
    dists, _ = tree.query(grid)
    
    # Convert Euclidean distances to angular distances
    # For unit vectors, angular distance = arccos(dot_product)
    # But we can use: angle = 2 * arcsin(d / 2) where d is Euclidean distance
    # This is more numerically stable
    max_d = np.max(dists)
    if max_d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(max_d / 2.0, -1.0, 1.0))


def _objective(x):
    """Objective function: covering radius in radians. Minimize this."""
    points = x.reshape(N, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points = points / norms
    
    return _covering_radius(points)


def _initial_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer
    
    # Initial points
    points = _initial_points()
    
    # Precompute grid for objective function
    grid = _spherical_grid(n_lat=100, n_lon=200)
    
    def obj_with_grid(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
        tree = cKDTree(pts)
        dists, _ = tree.query(grid)
        max_d = np.max(dists)
        if max_d >= 2.0:
            return np.pi
        return 2.0 * np.arcsin(np.clip(max_d / 2.0, -1.0, 1.0))
    
    # Try to optimize using L-BFGS-B
    best_points = points.copy()
    best_radius = _covering_radius(points, grid)
    
    try:
        x0 = points.flatten()
        
        # Use L-BFGS-B optimization
        result = minimize(
            obj_with_grid,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < timeout:
            opt_points = result.x.reshape(N, 3)
            norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            opt_points = opt_points / norms
            
            opt_radius = _covering_radius(opt_points, grid)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
        
        # Try a second round of optimization from the best point found
        if time.time() - start_time < timeout:
            x0 = best_points.flatten()
            result2 = minimize(
                obj_with_grid,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() - start_time < timeout:
                opt_points2 = result2.x.reshape(N, 3)
                norms = np.linalg.norm(opt_points2, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                opt_points2 = opt_points2 / norms
                
                opt_radius2 = _covering_radius(opt_points2, grid)
                if opt_radius2 < best_radius:
                    best_points = opt_points2
                    best_radius = opt_radius2
                    
    except Exception:
        pass
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_points = best_points / norms
    
    # Check for coincident points and hemisphere issues
    # If points are too close or all in one hemisphere, fall back to initial
    tree = cKDTree(best_points)
    dists, _ = tree.query(best_points, k=2)
    if np.min(dists[:, 1]) < 1e-6:
        best_points = _initial_points()
    
    # Check if all points are in a closed hemisphere
    # This is a heuristic: check if there's a direction with all dots positive
    # For a proper check, we'd need to solve a linear program, but this is a quick heuristic
    if time.time() - start_time < timeout:
        # Simple check: if the centroid is near the origin, likely not in a hemisphere
        centroid = np.mean(best_points, axis=0)
        if np.linalg.norm(centroid) > 0.8:
            # Might be in a hemisphere, try to fix by adding random perturbation
            pass
    
    return best_points
# EVOLVE-BLOCK-END
