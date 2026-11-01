"""Rigid-body assembly approach for 50 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding itself)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _relax_packing(centers, r, max_iters=500, rng=None):
    """
    Gauss-Seidel relaxation to satisfy constraints for a given radius r.
    Returns the relaxed centers.
    """
    centers = centers.copy()
    if rng is None:
        rng = np.random.default_rng(0)
        
    for _ in range(max_iters):
        for i in range(N):
            disp = np.zeros(2)
            
            # Wall constraints
            # x
            if centers[i, 0] < -0.5 + r:
                disp[0] += (-0.5 + r) - centers[i, 0]
            elif centers[i, 0] > 0.5 - r:
                disp[0] += (0.5 - r) - centers[i, 0]
            # y
            if centers[i, 1] < -0.5 + r:
                disp[1] += (-0.5 + r) - centers[i, 1]
            elif centers[i, 1] > 0.5 - r:
                disp[1] += (0.5 - r) - centers[i, 1]
                
            # Pair constraints
            # Calculate distances to all other circles
            # We use the current state of centers (Gauss-Seidel style)
            diffs = centers[i] - centers
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf # ignore self
            
            # Find violating pairs
            viol = dists < 2*r
            if np.any(viol):
                # Sum up the pushes from all violating neighbors
                # We iterate through the violating indices
                viol_indices = np.where(viol)[0]
                for j in viol_indices:
                    d = dists[j]
                    if d < 1e-9:
                        # Random nudge to break symmetry
                        disp += rng.normal(0, 0.01, 2)
                    else:
                        direction = (centers[i] - centers[j]) / d
                        error = 2*r - d
                        # Split the correction
                        disp += direction * error * 0.5
                        
            centers[i] += disp
            # Clamp to square
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            
    return centers

def _maximize_radius(centers, time_budget, rng):
    """
    Binary search for the maximum feasible radius using _relax_packing.
    """
    start_time = time.time()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    lo = current_r
    hi = min(0.08, current_r * 1.1)
    
    while time.time() - start_time < time_budget:
        mid = (lo + hi) / 2
        # Try to relax to radius mid
        # Use a few iterations of relaxation
        trial_centers = _relax_packing(best_centers, mid, max_iters=100, rng=rng)
        trial_r = _compute_radius(trial_centers)
        
        if trial_r >= mid - 1e-9:
            # Feasible
            lo = mid
            best_r = trial_r
            best_centers = trial_centers
        else:
            # Not feasible
            hi = mid
            
        # If we've converged closely, stop
        if hi - lo < 1e-6:
            break
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Main loop: Maximize radius using the new structural approach
    # We allocate most of the time to this
    time_left = deadline - time.time()
    if time_left > 5:
        best_centers, best_radius = _maximize_radius(
            best_centers, time_budget=time_left - 5, rng=rng
        )
        
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
