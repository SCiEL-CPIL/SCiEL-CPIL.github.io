"""Optimized program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time


def _spherical_grid(n_points=20000):
    """Generate a dense spherical grid using Fibonacci spiral."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius_exact(points, grid=None):
    """Compute the exact covering radius in radians using a dense grid."""
    if grid is None:
        grid = _spherical_grid()
    
    # Find nearest neighbor for each grid point
    tree = cKDTree(points)
    dists, _ = tree.query(grid)
    
    # Convert Euclidean distances to angular distances
    max_d = np.max(dists)
    if max_d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(max_d / 2.0, -1.0, 1.0))


def _smooth_objective(x, grid, temperature=0.1):
    """
    Smooth, differentiable objective function using soft-min of squared distances.
    This creates a smooth landscape for gradient-based optimization.
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    pts = pts / norms
    
    # Compute squared Euclidean distances from each grid point to each sphere point
    dots = grid @ pts.T  # (M, N)
    dist_sq = 2.0 - 2.0 * dots  # (M, N)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Soft min over points for each grid point (nearest point)
    scaled = -dist_sq / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_scaled.squeeze() + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
    soft_min_dist_sq = -temperature * log_sum_exp
    
    # Smooth max over grid points
    temp2 = temperature * 0.5
    scaled2 = -soft_min_dist_sq / temp2
    max_scaled2 = np.max(scaled2)
    log_sum_exp2 = max_scaled2 + np.log(np.sum(np.exp(scaled2 - max_scaled2)))
    smooth_max_dist_sq = -temp2 * log_sum_exp2
    
    # Convert squared distance to angular distance
    d = np.sqrt(np.maximum(smooth_max_dist_sq, 0.0))
    if d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(d / 2.0, -1.0, 1.0))


def _initial_points(offset=0.0):
    """Generate initial points using Fibonacci spiral with optional offset."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_refinement(points, grid, max_iter=200, step_size=0.01):
    """Local refinement using gradient-free search with exact covering radius."""
    best_points = points.copy()
    best_radius = _covering_radius_exact(best_points, grid)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for dim in range(3):
                # Try small perturbations in each direction
                for delta in [step_size, -step_size]:
                    new_points = best_points.copy()
                    new_points[i, dim] += delta
                    # Re-normalize
                    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    new_points = new_points / norms
                    
                    new_radius = _covering_radius_exact(new_points, grid)
                    if new_radius < best_radius - 1e-10:
                        best_points = new_points
                        best_radius = new_radius
                        improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer
    
    # Precompute grid for objective function
    grid = _spherical_grid(n_points=20000)
    
    best_points = None
    best_radius = np.pi
    
    # Try multiple initial configurations
    offsets = [0.0, 0.3, 0.7, 1.2, 2.1]
    
    for offset in offsets:
        if time.time() - start_time > timeout:
            break
        
        points = _initial_points(offset)
        current_radius = _covering_radius_exact(points, grid)
        
        if current_radius < best_radius:
            best_points = points.copy()
            best_radius = current_radius
        
        # Phase 1: Optimize with smooth objective using annealed temperature
        temperatures = [0.3, 0.15, 0.08, 0.04, 0.02]
        
        for temp in temperatures:
            if time.time() - start_time > timeout:
                break
            
            def obj_smooth(x, _grid=grid, _temp=temp):
                return _smooth_objective(x, _grid, temperature=_temp)
            
            x0 = points.flatten()
            
            try:
                result = minimize(
                    obj_smooth,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
            except Exception:
                break
            
            if time.time() - start_time > timeout:
                break
            
            opt_points = result.x.reshape(N, 3)
            norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            opt_points = opt_points / norms
            
            opt_radius = _covering_radius_exact(opt_points, grid)
            if opt_radius < current_radius:
                points = opt_points
                current_radius = opt_radius
        
        # Phase 2: Local refinement
        if time.time() - start_time < timeout:
            refined_points, refined_radius = _local_refinement(points, grid, max_iter=50, step_size=0.005)
            if refined_radius < current_radius:
                points = refined_points
                current_radius = refined_radius
        
        if current_radius < best_radius:
            best_points = points.copy()
            best_radius = current_radius
    
    if best_points is None:
        best_points = _initial_points()
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_points = best_points / norms
    
    # Check for coincident points and hemisphere issues
    tree = cKDTree(best_points)
    dists, _ = tree.query(best_points, k=2)
    if np.min(dists[:, 1]) < 1e-6:
        best_points = _initial_points()
    
    return best_points
# EVOLVE-BLOCK-END
