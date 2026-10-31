"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _score(points):
    # points: (N, 2)
    # Center and scale
    pts = points - np.mean(points, axis=0)
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-10:
        return 1e10
    pts = pts / rms
    
    # Pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Exclude self-distances
    np.fill_diagonal(dists_sq, np.inf)
    
    min_d2 = np.min(dists_sq)
    max_d2 = np.max(dists_sq)
    
    if min_d2 < 1e-10:
        return 1e10
        
    return max_d2 / min_d2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: 4x4 grid with radial scaling
    # Create a 4x4 grid
    x_grid, y_grid = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    pts = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    
    # Apply "square-root radial" displacement
    # Calculate distance from center for each point
    radii = np.linalg.norm(pts, axis=1)
    # Avoid division by zero (center is not in 4x4 grid, min radius is > 0)
    # Scale radii by a factor based on index to create non-uniform distribution
    # This helps mitigate corner-center disparity by pushing outer points out slightly
    # and pulling inner points in, or vice versa depending on the factor.
    # Here we use sqrt(index) as a proxy for radial ordering, though grid order is row-major.
    # A better approach: sort by radius, apply sqrt scaling, then map back?
    # For simplicity and robustness, we just use the grid points. The optimizer will adjust.
    # To strictly follow the hint, we scale the radius of each point by (1 + 0.1 * sqrt(i))
    indices = np.arange(N)
    scale = 1.0 + 0.1 * np.sqrt(indices)
    pts[:, 0] *= scale
    pts[:, 1] *= scale
        
    x0 = pts.flatten()
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Differential Evolution
    bounds = [(-10, 10)] * (N * 2)
    
    try:
        res = differential_evolution(
            lambda x: _score(x.reshape(N, 2)), 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-4, 
            update_interval=5, 
            callback=lambda xk, convergence: (time.time() > deadline)
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, 2)
            
        # Polish with Nelder-Mead
        if time.time() < deadline - 10:
            res2 = minimize(
                lambda x: _score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_pts = res2.x.reshape(N, 2)
                
    except Exception:
        pass
        
    return best_pts
# EVOLVE-BLOCK-END
