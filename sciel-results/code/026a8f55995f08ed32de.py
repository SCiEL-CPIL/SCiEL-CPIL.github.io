"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calc_radius(pts):
    """Calculate the maximum radius for a given set of points."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distances to walls
    # Square is [-0.5, 0.5]
    dists_wall = np.minimum(np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]), 
                            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1]))
    min_wall = np.min(dists_wall)
    
    # Radius is limited by half the minimum pair distance OR the minimum wall distance
    return min(min_pair / 2.0, min_wall)

def _energy(x, beta=50.0):
    """
    Energy function to minimize.
    Uses inverse power repulsion with soft-min for smoother optimization.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0) # Avoid division by zero on diagonal
    
    # Wall distances
    dists_wall = np.minimum(np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]), 
                            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1]))
    
    # Clip to avoid infinity
    dists = np.maximum(dists, 1e-8)
    dists_wall = np.maximum(dists_wall, 1e-8)
    
    # Use higher power for stronger repulsion at close range
    e_pairs = np.sum(1.0 / (dists ** 3))
    e_walls = np.sum(1.0 / (dists_wall ** 3))
    
    return e_pairs + e_walls

def _hex_grid_init(seed=0):
    """Create hexagonal grid initialization for better circle packing."""
    np.random.seed(seed)
    
    # Hexagonal grid: rows offset by half spacing
    # For 200 circles in unit square, we need ~14x14 grid
    m = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))  # Account for hex packing density
    
    # Hex spacing
    dx = 1.0 / (m + 1)
    dy = dx * np.sqrt(3)
    
    points = []
    for row in range(m + 2):
        y = -0.5 + dy * (row + 0.5)
        if y > 0.5:
            break
        # Offset every other row
        offset = (row % 2) * (dx / 2)
        for col in range(m + 2):
            x = -0.5 + dx * (col + 0.5) + offset
            if -0.5 <= x <= 0.5:
                points.append([x, y])
    
    # Convert to array
    points = np.array(points)
    
    # If we have too many points, randomly select N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    # If we have too few, add random points
    while len(points) < N:
        x = np.random.uniform(-0.45, 0.45)
        y = np.random.uniform(-0.45, 0.45)
        points = np.vstack([points, [[x, y]]])
    
    # Add small random jitter to break symmetry
    points += np.random.uniform(-0.005, 0.005, points.shape)
    
    # Ensure points are strictly inside bounds
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _local_refine(x, max_iter=5000):
    """Perform local refinement with tighter tolerances."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(_energy, x, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
        return res.x.reshape(N, 2)
    except Exception:
        return x

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s safety margin for 150s limit
    
    best_x = None
    best_r = 0.0
    
    # Number of restarts - use more diverse initializations
    n_restarts = 25
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Use hexagonal grid with different seeds for diversity
        x0 = _hex_grid_init(seed=i * 7 + 13)
        
        try:
            # First optimization pass
            res = minimize(_energy, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 40})
            x_opt = res.x.reshape(N, 2)
            
            # Local refinement pass with tighter tolerances
            x_opt = _local_refine(x_opt, max_iter=2000)
            
            r = _calc_radius(x_opt)
            
            if r > best_r:
                best_r = r
                best_x = x_opt.copy()
                
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return hex grid
        best_x = _hex_grid_init(seed=42)
        
    return best_x
# EVOLVE-BLOCK-END
