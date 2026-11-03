"""Voronoi-based packing optimization for 50 equal circles in a unit square."""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _voronoi_inradii(centers):
    """
    Compute inradius for each circle based on Voronoi diagram.
    The inradius is the minimum distance from center to cell boundary.
    For interior cells, this is the inradius of the Voronoi cell.
    For boundary cells, we also account for the square walls.
    """
    n = len(centers)
    inradii = np.full(n, np.inf)
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback: use pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        inradii = np.min(dists, axis=1) / 2.0
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        inradii = np.minimum(inradii, wall_dists)
        return inradii
    
    # For each point, find the minimum distance to any Voronoi edge
    # The Voronoi vertices are the circumcenters of Delaunay triangles
    # The inradius is the min distance from center to any point on cell boundary
    
    # Method: for each point, the inradius is the min distance to:
    # 1. Any Voronoi vertex (circumcenter) that belongs to this point's cell
    # 2. The walls of the square (for boundary points)
    
    # Get Voronoi vertices and regions
    vor_vertices = vor.vertices
    vor_pointregions = vor.pointregion
    
    for i in range(n):
        region = vor_pointregions[i]
        if region is None:
            inradii[i] = 0.0
            continue
        
        min_dist = np.inf
        
        # Check distances to Voronoi vertices in this region
        for vi in region:
            if vi < 0:
                # Unbounded vertex (infinity)
                continue
            v = vor_vertices[vi]
            d = np.linalg.norm(centers[i] - v)
            if d < min_dist:
                min_dist = d
        
        # Check wall distances
        wall_d = np.min(0.5 - np.abs(centers[i]))
        if wall_d < min_dist:
            min_dist = wall_d
        
        # If no finite vertices found (unbounded cell), rely on walls
        if np.isinf(min_dist):
            min_dist = wall_d
            
        inradii[i] = min_dist
    
    return inradii

def _voronoi_objective(params):
    """
    Objective function to maximize minimum inradius.
    Uses soft-min to make it differentiable.
    """
    centers = params.reshape(N, 2)
    centers = np.clip(centers, -0.4999, 0.4999)
    
    inradii = _voronoi_inradii(centers)
    
    if len(inradii) == 0:
        return 0.0
    
    # Soft min
    k = 500.0
    m = np.min(inradii)
    shifted = inradii - m
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min = m - (1.0 / k) * log_sum
    return -soft_min

def _optimize_from_start(start_centers, bounds, max_time=20.0):
    """
    Optimization using Voronoi-based objective.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    # First pass
    try:
        result = minimize(
            _voronoi_objective,
            best_centers.reshape(-1),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.4999, 0.4999)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
    
    # Second pass with tighter tolerances
    try:
        result = minimize(
            _voronoi_objective,
            best_centers.reshape(-1),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.4999, 0.4999)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
    
    return best_centers, best_score

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal packing
    s = 0.142
    points = []
    for i, x in enumerate(np.arange(-0.8, 0.8, s)):
        offset = (s * np.sqrt(3) / 2) if i % 2 == 0 else 0
        for y in np.arange(-0.8, 0.8, s):
            points.append((x, y + offset))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.48) & (np.abs(points[:, 1]) <= 0.48)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
    
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 8)
    grid_y = np.linspace(-0.45, 0.45, 8)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
    
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.015, (N, 2))
        jittered = np.clip(jittered, -0.48, 0.48)
        inits.append(jittered)
    
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.48, 0.48, 8))
    grid_y_rand = np.sort(np.random.uniform(-0.48, 0.48, 8))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])
    
    # 7. Symmetric-ish random with reflections
    np.random.seed(777)
    half = np.random.uniform(-0.45, 0.45, (N//2, 2))
    mirrored = np.column_stack([half[:, 0], half[:, 1]])
    mirrored2 = np.column_stack([-half[:, 0], -half[:, 1]])
    inits.append(np.vstack([mirrored, mirrored2]))
    
    return inits

def construct():
    bounds = [(-0.4999, 0.4999)] * (2 * N)
    
    inits = _generate_initializations()
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > 100.0:
            break
            
        time_limit = 18.0
        if elapsed + time_limit > 110.0:
            time_limit = max(5.0, 110.0 - elapsed)
        
        centers, score = _optimize_from_start(init, bounds, max_time=time_limit)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
