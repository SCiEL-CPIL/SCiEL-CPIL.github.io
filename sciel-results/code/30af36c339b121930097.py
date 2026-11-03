"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

def _spherical_centroid(pts, cell_points):
    """
    Compute the spherical centroid of a set of points on the unit sphere.
    Uses the normalized sum of vectors weighted by solid angle approximation.
    """
    if len(cell_points) == 0:
        return None
    if len(cell_points) == 1:
        return cell_points[0].copy()
    
    # Weight by the angle subtended at the center
    # For a Voronoi cell, the centroid is approximately the normalized sum
    # of the vertex vectors, but better to use solid angle weights
    
    # Simple approach: normalize the sum of all vertex vectors
    # This gives the "mean direction" which is a good approximation for small cells
    vec_sum = np.sum(cell_points, axis=0)
    norm = np.linalg.norm(vec_sum)
    if norm < 1e-10:
        # Degenerate case: use uniform average direction
        vec_sum = np.mean(cell_points, axis=0)
        norm = np.linalg.norm(vec_sum)
        if norm < 1e-10:
            return cell_points[0].copy()
    return vec_sum / norm

def _compute_voronoi_centroids(pts, n_iter=1):
    """
    Compute Voronoi cell centroids for points on the sphere using
    Delaunay triangulation in 3D (projected).
    
    Returns array of centroids, one per input point.
    """
    N = len(pts)
    centroids = np.zeros_like(pts)
    
    for i in range(N):
        # Find all points that are "neighbors" of pts[i] in the Delaunay sense
        # For now, use a simpler approach: find the k-nearest neighbors
        # and compute their spherical centroid
        pass
    
    return centroids

def _lloyd_iteration(pts, grid, alpha=1.0):
    """
    Perform one Lloyd iteration: move each point to the centroid of its
    spherical Voronoi cell.
    
    We approximate the Voronoi cell by finding the region of the grid
    where each point is the nearest neighbor, then computing the spherical
    centroid of that region.
    """
    N = len(pts)
    M = grid.shape[0]
    
    # Find nearest point for each grid point
    dots = grid @ pts.T  # (M, N)
    nearest = np.argmax(dots, axis=1)  # (M,)
    
    # For each point, collect the grid points in its Voronoi cell
    centroids = np.zeros_like(pts)
    
    for i in range(N):
        mask = (nearest == i)
        if np.sum(mask) > 0:
            cell_pts = grid[mask]
            # Spherical centroid: normalize the sum
            vec_sum = np.sum(cell_pts, axis=0)
            norm = np.linalg.norm(vec_sum)
            if norm > 1e-10:
                centroids[i] = vec_sum / norm
            else:
                centroids[i] = pts[i]
        else:
            centroids[i] = pts[i]
    
    # Move points toward their centroids
    new_pts = pts + alpha * (centroids - pts)
    # Normalize back to unit sphere
    new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
    
    return new_pts

def _compute_covering_radius(pts, grid):
    """Compute the covering radius (max angular distance) using grid."""
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    return np.max(angles)

def _lloyd_optimize(pts, grid, time_limit, max_iter=2000):
    """
    Optimize points using Lloyd's algorithm (Voronoi relaxation).
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts, grid)
    
    alpha = 1.0
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perform one Lloyd iteration
        new_pts = _lloyd_iteration(current_pts, grid, alpha=alpha)
        
        # Check for improvement
        new_radius = _compute_covering_radius(new_pts, grid)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_pts = new_pts.copy()
            no_improve = 0
            # Increase alpha slightly to speed up convergence
            alpha = min(alpha * 1.02, 1.0)
        elif new_radius < current_radius if 'current_radius' in dir() else False:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Try smaller alpha
                alpha *= 0.5
                if alpha < 0.01:
                    break
                no_improve = 0
        
        current_pts = new_pts
        current_radius = new_radius
    
    return best_pts, best_radius

def _generate_grid(n_points=20000):
    """Generate a dense set of points on the sphere for discretization."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    # Uniform distribution on sphere
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    total_time_limit = 130.0  # Safety margin before 150s kill
    
    # Use a dense grid for accurate covering radius estimation
    grid = _generate_grid(20000)
    
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.RandomState(42)
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # 2-5. Random initializations
    for _ in range(4):
        initializations.append(_random_sphere(N, rng))
    
    # 6. Perturbed Fibonacci
    fib = _fibonacci_sphere(N)
    perturbed = fib + rng.normal(0, 0.1, fib.shape)
    perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
    initializations.append(perturbed)
    
    time_per_init = (total_time_limit - 5.0) / len(initializations)
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - start_time > total_time_limit - 10:
            break
        
        time_left = min(time_per_init, total_time_limit - (time.time() - start_time) - 5)
        if time_left < 2:
            continue
        
        # Run Lloyd optimization
        result_pts, result_rad = _lloyd_optimize(init_pts, grid, time_limit=time_left)
        
        # Verify with a finer grid for final score
        fine_grid = _generate_grid(50000)
        fine_rad = _compute_covering_radius(result_pts, fine_grid)
        
        if fine_rad < best_radius:
            best_radius = fine_rad
            best_pts = result_pts.copy()
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
