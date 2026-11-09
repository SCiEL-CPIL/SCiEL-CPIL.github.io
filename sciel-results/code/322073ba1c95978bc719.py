"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_radius(points):
    """
    Compute the achievable common radius for given centers.
    Returns the minimum of:
    1. Half the minimum distance between any pair of points.
    2. The minimum distance from any point to the boundary (1 - norm).
    """
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Boundary constraint: distance to edge
    norms = np.linalg.norm(points, axis=1)
    boundary_min = np.min(1.0 - norms)
    
    # Pairwise distances
    # Vectorized pairwise distance calculation
    # diff shape: (N, N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # We only care about i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[i_idx, j_idx]
    
    if len(pair_dists_sq) == 0:
        pair_min = np.inf
    else:
        pair_min = np.min(pair_dists_sq)
        
    # The radius is limited by half the minimum distance
    r_pair = 0.5 * np.sqrt(pair_min) if pair_min < np.inf else 1.0
    r_boundary = boundary_min
    
    return min(r_pair, r_boundary)

def _objective(x):
    """
    Objective function for minimization.
    We want to maximize the radius, so we minimize the negative radius.
    """
    points = x.reshape(N, 2)
    r = _compute_radius(points)
    return -r

def _initial_hex_grid(spacing_factor=1.0):
    """
    Generate initial hexagonal grid points.
    spacing_factor allows tuning the density.
    """
    # Target radius is approx 0.125, so diameter 0.25.
    # Hex grid spacing should be slightly larger than diameter to avoid initial overlaps,
    # but close enough to be a good starting point.
    base_spacing = 0.26 * spacing_factor
    
    rows = []
    # Generate a grid large enough to cover the unit circle
    k = int(1.0 / base_spacing) + 3
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * base_spacing
            y = j * base_spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    # Filter points inside the circle (with a small margin to ensure they can move)
    # We want them strictly inside so the optimizer has room.
    mask = radii <= 1.0 - base_spacing / 2.0 - 0.01
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < N:
        # If not enough points, fill with random points near the center
        # This shouldn't happen with the grid above, but just in case
        while len(points) < N:
            p = np.random.randn(2) * 0.5
            points = np.vstack([points, p])
    
    # Select the N points closest to the center to form a compact cluster
    idx = np.argsort(radii, kind='stable')[:N]
    return points[idx]

def _nelder_mead_optimize(x0, time_limit=10.0):
    """
    Run Nelder-Mead optimization with a time limit.
    """
    start_time = time.time()
    
    # Nelder-Mead options
    # maxiter is a hard stop, but we also check time
    # initial_simplex_size and step_size can be tuned
    options = {
        'maxiter': 2000,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'adaptive': True,
        'initial_simplex_size': 0.01, # Small initial step
    }
    
    # We need a wrapper to check time
    # scipy's minimize doesn't support callback for time limits directly in all versions efficiently for stopping,
    # so we rely on maxiter and the fact that Nelder-Mead is relatively fast per iteration.
    # However, 2000 iterations might take too long if we do multiple restarts.
    # Let's reduce maxiter and rely on multiple short runs or one long run.
    
    try:
        result = minimize(
            _objective, 
            x0, 
            method='Nelder-Mead', 
            options=options
        )
        return result.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (10s buffer)
    
    # 1. Initial Hex Grid
    # Try a few different spacing factors to see which gives a better start
    best_points = None
    best_radius = -np.inf
    
    # Standard hex grid
    points0 = _initial_hex_grid(1.0)
    r0 = _compute_radius(points0)
    best_points = points0
    best_radius = r0
    
    # 2. Nelder-Mead Optimization
    # We will run Nelder-Mead in a loop, potentially with perturbations (restarts)
    # if we have time.
    
    current_points = best_points.copy()
    
    # Run 1: Direct optimization
    if time.time() < deadline - 20:
        x0 = current_points.flatten()
        x_opt = _nelder_mead_optimize(x0)
        new_points = x_opt.reshape(N, 2)
        new_radius = _compute_radius(new_points)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_points = new_points
            current_points = new_points
    
    # Run 2-5: Perturbed restarts (Simulated Annealing lite)
    # If we are stuck, we might try shaking the points slightly and re-optimizing.
    # This helps escape local minima where points are "locked" in a suboptimal configuration.
    
    for i in range(10): # Max 10 restarts, but limited by time
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
            
        # Generate a perturbed version of the current best
        # Perturbation magnitude decreases as we iterate (cooling)
        # Or we can just use a fixed small perturbation
        perturbation_scale = 0.05 * (1.0 / (1.0 + i))
        
        # Create a new starting point by adding noise
        noise = np.random.randn(N, 2) * perturbation_scale
        perturbed_points = best_points + noise
        
        # Ensure perturbed points are inside the circle
        # If a point goes outside, project it back or just let the optimizer fix it?
        # Nelder-Mead can handle boundary violations if the objective penalizes them (which it does via r_boundary becoming negative or small).
        # However, _compute_radius returns min(r_pair, r_boundary). If r_boundary is negative, the score is negative.
        # It's better to start inside.
        norms = np.linalg.norm(perturbed_points, axis=1)
        mask_out = norms > 0.95
        if np.any(mask_out):
            # Scale back points that are too close to edge
            scale_factors = np.where(norms > 0.95, 0.95 / norms, 1.0)
            perturbed_points *= scale_factors[:, np.newaxis]
            
        x0 = perturbed_points.flatten()
        
        # Run a shorter optimization for restarts to save time
        # We can reuse the helper but with fewer iterations
        # Let's just call the same helper, it's fast enough usually
        x_opt = _nelder_mead_optimize(x0)
        new_points = x_opt.reshape(N, 2)
        new_radius = _compute_radius(new_points)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_points = new_points
            
            # If we improved, we might want to continue from this new best
            # rather than the original best for the next perturbation?
            # Yes, hill climbing.
    
    # Final Validation
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(~np.isfinite(best_points)) or np.any(norms > 1.0 + 1e-9):
        # Fallback to initial grid if something went wrong
        best_points = _initial_hex_grid(1.0)
        
    return best_points
# EVOLVE-BLOCK-END
