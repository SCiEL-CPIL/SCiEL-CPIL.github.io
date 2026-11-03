"""Improved program using Voronoi-diagram-based optimization with SA polishing."""
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
    Compute the inradius of each Voronoi cell (distance from center to nearest cell edge).
    Returns array of shape (N,) with inradius for each circle.
    """
    # Add boundary points to ensure all Voronoi cells are bounded
    # We'll use a large bounding box
    eps = 0.01
    boundary = np.array([
        [-0.5-eps, -0.5-eps], [-0.5-eps, 0.5+eps],
        [0.5+eps, -0.5-eps], [0.5+eps, 0.5+eps],
        [-0.5-eps, 0.0], [0.5+eps, 0.0],
        [0.0, -0.5-eps], [0.0, 0.5+eps]
    ])
    
    all_points = np.vstack([centers, boundary])
    
    try:
        vor = Voronoi(all_points)
    except Exception:
        # Fallback: use pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1) / 2.0
        wall_dists = 0.5 - np.abs(centers)
        min_walls = np.min(wall_dists, axis=1)
        return np.minimum(min_dists, min_walls)
    
    inradii = np.zeros(N)
    
    for i in range(N):
        # Get the Voronoi region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region - use wall distance
            wall_dists = 0.5 - np.abs(centers[i])
            inradii[i] = np.min(wall_dists)
            continue
        
        # Get the vertices of this cell
        vertices = vor.vertices[region]
        
        # The inradius is the minimum distance from center to any edge of the cell
        # For a convex polygon, the inradius is the minimum distance from the center to any edge
        # We can compute this by finding the minimum distance from center to each edge
        
        min_dist = np.inf
        n_verts = len(vertices)
        
        for j in range(n_verts):
            v1 = vertices[j]
            v2 = vertices[(j+1) % n_verts]
            
            # Distance from point to line segment
            edge_vec = v2 - v1
            center_vec = centers[i] - v1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            
            if edge_len_sq < 1e-15:
                dist = np.linalg.norm(center_vec)
            else:
                t = np.dot(center_vec, edge_vec) / edge_len_sq
                t = np.clip(t, 0, 1)
                closest = v1 + t * edge_vec
                dist = np.linalg.norm(centers[i] - closest)
            
            min_dist = min(min_dist, dist)
        
        inradii[i] = min_dist
    
    return inradii

def _optimize_voronoi(centers, bounds):
    """
    Optimize using a smooth objective based on Voronoi inradii.
    """
    best_centers = centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    def objective(params):
        c = params.reshape(N, 2)
        c = np.clip(c, -0.499, 0.499)
        inradii = _voronoi_inradii(c)
        # Use a soft min to make it differentiable
        k = 500.0
        m = np.min(inradii)
        shifted = inradii - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min
    
    # First pass
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        centers_new = result.x.reshape(N, 2)
        centers_new = np.clip(centers_new, -0.499, 0.499)
        true_r = _calculate_true_radius(centers_new)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers_new
    except Exception:
        pass
    
    # Second pass with higher precision
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        centers_new = result.x.reshape(N, 2)
        centers_new = np.clip(centers_new, -0.499, 0.499)
        true_r = _calculate_true_radius(centers_new)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers_new
    except Exception:
        pass
    
    return best_centers, best_score

def _simulated_annealing(centers, bounds, max_iters=30000, start_temp=0.005, end_temp=1e-7):
    """
    Simulated Annealing to escape local minima.
    """
    centers = centers.copy()
    current_score = _calculate_true_radius(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    n_params = len(centers) * 2
    
    for it in range(max_iters):
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Generate a perturbation
        step_size = temp * 2
        perturbation = np.random.normal(0, step_size, n_params)
        new_params = centers.reshape(-1) + perturbation
        new_centers = np.clip(new_params.reshape(N, 2), -0.499, 0.499)
        
        new_score = _calculate_true_radius(new_centers)
        
        # Accept if better or with probability exp((new - old) / temp)
        if new_score >= current_score:
            centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / max(temp, 1e-10)):
                centers = new_centers
                current_score = new_score
    
    return best_centers, best_score

def construct():
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    s = 0.14 
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
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
        jittered = points[indices] + np.random.normal(0, 0.02, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 90.0: 
            break
        
        # First, optimize with Voronoi-based method
        centers, score = _optimize_voronoi(init, bounds)
        
        if score > best_score:
            best_score = score
            best_centers = centers
        
        # Then, polish with SA
        if time.time() - start_time < 105.0:
            sa_centers, sa_score = _simulated_annealing(
                centers, bounds, 
                max_iters=15000, 
                start_temp=0.003, 
                end_temp=1e-8
            )
            if sa_score > best_score:
                best_score = sa_score
                best_centers = sa_centers
    
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
