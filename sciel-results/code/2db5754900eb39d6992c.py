import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _soft_min_radius(points, k):
    """
    Calculate the soft-min radius for a set of points.
    points: (N, 2) array of centers.
    k: temperature parameter for soft-min. Higher k -> closer to hard min.
    """
    # Distances between all pairs
    # Use broadcasting to compute distance matrix
    # diff[i, j, :] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Boundary distances: 1 - |points|
    dists_bound = 1.0 - np.sqrt(np.sum(points**2, axis=-1))
    
    # We want min( min(i<j) dists[i,j]/2, min(i) dists_bound[i] )
    # Mask out diagonal and double count for pairs
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dists[mask] / 2.0
    
    # Combine pair distances and boundary distances
    all_dists = np.concatenate([pair_dists, dists_bound])
    
    # Soft min: -1/k * log(sum(exp(-k * d)))
    # To avoid overflow/underflow, subtract max
    max_d = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-k * (all_dists - max_d))))
    soft_r = -1.0 / k * (log_sum - k * max_d)
    
    return soft_r

def _objective(params, k):
    """
    Objective function for minimization.
    We want to maximize soft radius, so we return negative soft radius.
    """
    points = params.reshape(-1, 2)
    r = _soft_min_radius(points, k)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # 1. Generate initial hexagonal lattice
    # Estimate spacing for ~300 points in unit circle
    # Area of unit circle = pi. Area per point ~ pi/300.
    # Hex lattice density ~ 1/(sqrt(3)/2 * s^2) ? No, area per point in hex lattice is s^2 * sqrt(3)/2.
    # s^2 * sqrt(3)/2 ~ pi/300 => s ~ sqrt(2*pi / (300*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate grid
    k_grid = int(1.0 / s) + 2
    xs = np.arange(-k_grid, k_grid + 1) * s
    ys = np.arange(-k_grid, k_grid + 1) * s * np.sqrt(3) / 2
    
    # Create hex pattern
    grid_x, grid_y = np.meshgrid(xs, ys)
    # Offset odd rows
    rows = np.arange(-k_grid, k_grid + 1)
    offset = (s / 2) * (rows % 2)
    grid_x += offset[np.newaxis, :]
    
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Keep only points inside circle (with margin)
    r_pts = np.sqrt(np.sum(points**2, axis=1))
    points = points[r_pts < 1.0 - s/2]
    
    # If we have too many, pick the ones closest to center? Or just first N?
    # For packing, we want them spread out.
    # Let's just take the first N if we have more, but maybe random sampling or sorting by radius is better.
    # Sorting by radius from center ensures we fill the center first, which is good for packing.
    if len(points) > N:
        # Sort by distance to center
        r_pts = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(r_pts)
        points = points[idx[:N]]
    elif len(points) < N:
        # If not enough, generate more or adjust spacing
        # For now, just pad with random points near center? 
        # Better: reduce spacing and regenerate
        # This loop is simple, we assume s is good enough. 
        # If s is too large, we might have < N.
        # Let's add a fallback: if < N, add random points
        while len(points) < N:
            new_pt = np.random.uniform(-1, 1, size=2)
            if np.linalg.norm(new_pt) < 0.9:
                points = np.vstack([points, new_pt])
        
    points = points.astype(np.float64)
    params = points.flatten()
    
    # 2. Optimization
    # Annealing schedule for k
    k_values = [5, 10, 20, 50, 100]
    
    best_params = params.copy()
    best_score = -_soft_min_radius(best_params.reshape(-1, 2), k_values[-1])
    
    for k in k_values:
        if time.time() > deadline - 30: # Leave 30s for final steps
            break
            
        # Define objective
        obj = lambda x: _objective(x, k)
        
        # Optimization
        # L-BFGS-B is good for large N
        # We might need to provide bounds to keep points in circle? 
        # The soft-min handles the boundary penalty, but hard constraints can help stability.
        # However, L-BFGS-B with bounds might get stuck on the boundary. 
        # Let's try without bounds first, relying on the soft penalty.
        
        try:
            res = minimize(obj, params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            if res.success or res.fun < best_score:
                params = res.x
                # Update best
                current_score = -_soft_min_radius(params.reshape(-1, 2), k)
                if current_score > best_score:
                    best_score = current_score
                    best_params = params.copy()
        except Exception:
            pass
            
        # Add small random perturbation to escape local minima?
        # Maybe not necessary with L-BFGS-B and multiple k values.
        
    # 3. Final check and return
    # Calculate true radius to ensure validity
    final_points = best_params.reshape(-1, 2)
    
    # Verify constraints
    # 1. Inside circle
    dists_center = np.sqrt(np.sum(final_points**2, axis=1))
    r_boundary = np.min(1.0 - dists_center)
    
    # 2. Pairwise distances
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dists_pair = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists_pair, np.inf)
    r_pair = np.min(dists_pair) / 2.0
    
    r_final = min(r_boundary, r_pair)
    
    # If invalid (overlap or outside), shrink points towards center? 
    # The optimization should have prevented this, but just in case.
    if r_final <= 0:
        # Fallback to initial lattice
        final_points = points
        
    return final_points
