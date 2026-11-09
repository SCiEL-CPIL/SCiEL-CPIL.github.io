import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _hex_grid_init():
    """Generate a hexagonal grid initialization."""
    # Approximate spacing for N circles
    # Area of unit circle is pi. Area per circle ~ pi/N.
    # Side of hex cell ~ sqrt(pi/N / (sqrt(3)/2))
    # Let's just use a fine grid and pick the best N
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    # Sort by distance to center, take closest N
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]

def _random_init():
    """Generate a random initialization inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    x = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return x

def _compute_radius(x):
    """Compute the bottleneck radius for a given configuration x."""
    # Pairwise distances
    # Use broadcasting for efficiency
    # x shape (N, 2)
    # diff shape (N, N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distances to boundary
    dists_center = np.linalg.norm(x, axis=1)
    min_wall_dist = np.min(1.0 - dists_center)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _violation(x, r):
    """Calculate the total violation for a fixed radius r."""
    # x shape (N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise violations: max(0, 2r - dist)^2
    # We only care about pairs where dist < 2r
    # To avoid computing sqrt for all, we can compare squared distances
    # dist^2 < (2r)^2 => dist < 2r
    dists_sq = np.sum(diff * diff, axis=2)
    target_sq = (2 * r) ** 2
    # Mask for overlapping pairs
    mask = dists_sq < target_sq
    # For those, violation is (2r - dist)^2 = (2r - sqrt(dists_sq))^2
    # This is differentiable? No, sqrt is fine, but max is not.
    # However, for optimization, we can just sum the squared violations for the violating pairs.
    # If no pairs violate, return 0.
    if not np.any(mask):
        pair_violation = 0.0
    else:
        # Get the distances for violating pairs
        # We need the actual distances, not squared
        # dists is already computed
        violating_dists = dists[mask]
        # 2r - dist
        diffs = 2 * r - violating_dists
        pair_violation = np.sum(diffs ** 2)
    
    # Wall violations: max(0, |c| - (1-r))^2
    dists_center = np.linalg.norm(x, axis=1)
    target_wall = 1.0 - r
    wall_mask = dists_center > target_wall
    if not np.any(wall_mask):
        wall_violation = 0.0
    else:
        violating_dists_c = dists_center[wall_mask]
        diffs_c = violating_dists_c - target_wall
        wall_violation = np.sum(diffs_c ** 2)
        
    return pair_violation + wall_violation

def _optimize_for_radius(x_init, r, time_limit):
    """Try to find a configuration with radius at least r starting from x_init."""
    # Objective function to minimize
    def obj(x_flat):
        x = x_flat.reshape(N, 2)
        return _violation(x, r)
        
    # Initial guess
    x0 = x_init.flatten()
    
    # Use BFGS for robustness, or L-BFGS-B
    # BFGS is good for small N
    start_time = time.time()
    
    try:
        res = minimize(obj, x0, method='BFGS', options={'maxiter': 100, 'maxfun': 200, 'disp': False})
    except Exception:
        return x_init, False

    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return x_init, False
        
    x_opt = res.x.reshape(N, 2)
    violation = _violation(x_opt, r)
    
    # Consider it feasible if violation is very small
    if violation < 1e-6:
        return x_opt, True
    else:
        return x_opt, False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_x = _hex_grid_init()
    best_r = _compute_radius(best_x)
    
    # List of initializations
    inits = [best_x]
    for _ in range(5):
        inits.append(_random_init())
    
    # Binary search range
    # Lower bound: current best
    # Upper bound: theoretical max ~ 0.15
    r_low = best_r
    r_high = 0.15
    
    # We will try to improve r_low for each initialization
    # To save time, we will do a coarse binary search first
    
    for x_init in inits:
        if time.time() > deadline:
            break
            
        # Reset search bounds for this init? 
        # Or continue from global best?
        # Let's try to beat the global best.
        local_low = best_r
        local_high = r_high
        
        # Coarse search steps
        for _ in range(15): # ~33 binary search steps total if we do 2 passes?
             if time.time() > deadline:
                break
             
             if local_high - local_low < 1e-5:
                 break
                 
             mid = (local_low + local_high) / 2.0
             
             # Try to optimize for radius mid
             x_opt, feasible = _optimize_for_radius(x_init, mid, 5.0) # 5s per optimization step
            
             if feasible:
                 local_low = mid
                 # Update global best if improved
                 r_opt = _compute_radius(x_opt)
                 if r_opt > best_r:
                     best_r = r_opt
                     best_x = x_opt
             else:
                 local_high = mid
                 
    return best_x
