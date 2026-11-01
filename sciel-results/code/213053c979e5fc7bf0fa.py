import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _covering_radius(points, grid_points):
    # Compute the distance from each grid point to the nearest configuration point
    # points: (N, 3)
    # grid_points: (M, 3)
    # We want the max over grid_points of min over points of angular distance
    
    # Flatten points and grid_points for broadcasting
    # dist_sq[i, j] = ||grid_points[i] - points[j]||^2
    # = ||grid_points[i]||^2 + ||points[j]||^2 - 2 * grid_points[i] . points[j]
    # Since both are on the unit sphere, ||grid_points[i]||^2 = 1, ||points[j]||^2 = 1
    # dist_sq[i, j] = 2 - 2 * grid_points[i] . points[j]
    
    # Angular distance is arccos(1 - dist_sq/2) = arccos(grid_points[i] . points[j])
    # We can work with cosines to avoid arccos until the end
    
    dots = grid_points @ points.T  # (M, N)
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    
    # The nearest point corresponds to the maximum dot product (smallest angle)
    max_dots = np.max(dots, axis=1)  # (M,)
    
    # The angular distance is arccos(max_dots)
    # We want the maximum of these distances, which corresponds to the minimum of max_dots
    min_max_dot = np.min(max_dots)
    
    # The covering radius in radians is arccos(min_max_dot)
    return np.arccos(min_max_dot)

def _find_worst_point(points, grid_points):
    # Find the grid point that is farthest from the nearest configuration point
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    worst_idx = np.argmin(max_dots)
    return grid_points[worst_idx]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin for overhead
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    # Create a grid of points on the sphere for evaluating the covering radius
    # Use a spherical grid with resolution sufficient to find the worst point
    # 100x100 grid gives 10000 points, which is fast enough
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()
    
    grid_points = np.column_stack([
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid),
        np.cos(theta_grid)
    ])
    
    best_points = points.copy()
    best_radius = _covering_radius(points, grid_points)
    
    # Local search
    step_size = 0.1
    for iteration in range(1000):
        if time.time() > deadline - 5.0:
            break
        
        worst_point = _find_worst_point(points, grid_points)
        
        # Find the 3 nearest points to the worst point
        dots = points @ worst_point
        dots = np.clip(dots, -1.0, 1.0)
        nearest_indices = np.argsort(-dots)[:3]
        
        # Move these points towards the worst point
        # The direction to move is the projection of (worst_point - points[i]) onto the tangent plane
        for idx in nearest_indices:
            p = points[idx]
            # Direction towards worst point
            direction = worst_point - p
            # Project onto tangent plane: direction - (direction . p) * p
            dot_dir_p = np.dot(direction, p)
            direction_tangent = direction - dot_dir_p * p
            norm_dir = np.linalg.norm(direction_tangent)
            if norm_dir > 1e-10:
                direction_tangent = direction_tangent / norm_dir
                points[idx] = p + step_size * direction_tangent
        
        points = _normalize(points)
        
        current_radius = _covering_radius(points, grid_points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points
