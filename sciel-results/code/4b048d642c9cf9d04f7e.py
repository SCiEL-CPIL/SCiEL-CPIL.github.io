"""Program for packing 50 equal spheres in a unit cube; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses a BCC lattice initialization followed by local optimization to maximize
    the minimum distance between centers and to the cube boundaries.
    """
    np.random.seed(42)
    
    # Generate BCC lattice points in the unit cube [-0.5, 0.5]^3
    # BCC: points at corners (0,0,0) and body center (0.5,0.5,0.5) of each conventional cell
    # We'll use a grid of conventional cells and generate both corner and body-center points
    
    # Determine grid size: we need approximately 50 points
    # BCC has 2 points per conventional cell, so we need ~25 cells
    # Use 3x3x3 = 27 cells -> 54 points, then select 50
    
    n_cells = 3
    cell_size = 1.0 / n_cells
    offset = -0.5  # Start at -0.5
    
    bcc_points = []
    
    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            for k in range(n_cells + 1):
                # Corner points
                x = offset + i * cell_size
                y = offset + j * cell_size
                z = offset + k * cell_size
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    bcc_points.append([x, y, z])
    
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                # Body center points
                x = offset + (i + 0.5) * cell_size
                y = offset + (j + 0.5) * cell_size
                z = offset + (k + 0.5) * cell_size
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    bcc_points.append([x, y, z])
    
    bcc_points = np.array(bcc_points)
    
    # Remove any duplicate points (shouldn't happen, but be safe)
    bcc_points = np.unique(bcc_points, axis=0)
    
    # If we have more than N points, select N points using farthest-point sampling
    if len(bcc_points) > N:
        # Greedy farthest-point sampling
        selected = [np.random.randint(len(bcc_points))]
        while len(selected) < N:
            unselected = [i for i in range(len(bcc_points)) if i not in selected]
            best_idx = -1
            best_min_dist = -1
            for idx in unselected:
                dists = np.linalg.norm(bcc_points[idx] - bcc_points[selected], axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
        
        centers = bcc_points[selected]
    elif len(bcc_points) < N:
        # Pad with random points if needed
        centers = bcc_points.copy()
        while len(centers) < N:
            centers = np.vstack([centers, np.random.uniform(-0.45, 0.45, 3)])
    else:
        centers = bcc_points.copy()
    
    centers = centers[:N]
    
    # Define the objective function: maximize minimum radius
    def objective(x):
        """Return negative of the minimum sphere radius (to be minimized)."""
        pts = x.reshape(N, 3)
        
        # Check bounds
        if np.any(np.abs(pts) > 0.5 + 1e-9):
            return 1e10
        
        # Compute all pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Distance to boundaries
        dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
        min_bound_dist = np.min(dist_to_bound)
        
        # The radius is half of the minimum of pair distances and boundary distances
        r = 0.5 * min(min_pair_dist, min_bound_dist)
        
        return -r
    
    # Use Nelder-Mead for local optimization
    x0 = centers.flatten()
    
    # Run optimization with multiple restarts
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try several restarts with different perturbation scales
    for restart in range(8):
        if restart == 0:
            x_init = x0.copy()
        else:
            scale = 0.01 * (restart + 1)
            x_init = x0 + np.random.normal(0, scale, N * 3)
            x_init = np.clip(x_init.reshape(N, 3), -0.499, 0.499).flatten()
        
        result = minimize(
            objective,
            x_init,
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    
    # Final result
    result_centers = best_x.reshape(N, 3)
    
    # Ensure all points are strictly inside the cube
    result_centers = np.clip(result_centers, -0.4999, 0.4999)
    
    return result_centers
# EVOLVE-BLOCK-END
