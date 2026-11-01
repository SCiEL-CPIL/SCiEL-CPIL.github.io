"""
Optimized packing of 50 equal circles in a unit square.
Combines Binary Search on radius with a Force-Directed Feasibility Solver.
"""

import numpy as np
import time

N = 50
HALF = 0.5

def _score(c):
    """
    Compute the largest common radius for centres c (N,2).
    """
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Distance to walls
    min_bound = np.min(HALF - np.abs(c))
    
    return min(min_pair, min_bound)

def _hex_seed(r):
    """
    Generate a hexagonal lattice seed with spacing 2r.
    This provides a good initial configuration for the force-directed solver.
    """
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    # Generate points in a hexagonal pattern
    while y <= HALF - r + 1e-9:
        # Offset every other row to create hexagonal packing
        x_offset = 0.5 * dx if (row % 2) else 0.0
        x = -HALF + r + x_offset
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        # Select the N points closest to the center to maintain symmetry
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        # Fallback to grid if hex grid is too sparse (unlikely for r ~ 0.07)
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
        
    # Ensure points are strictly inside bounds for the solver
    return np.clip(pts, -HALF + r, HALF - r)

def _grid_seed(r):
    """Generate a standard grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)

def _feasible(pts, r, max_iter=500, deadline=None):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses force-directed relaxation (projected gradient descent).
    Returns (success, pts).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    # Adaptive step size
    step = 0.1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances and forces
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        
        # 1. Pairwise Repulsion Forces
        # We want to push apart pairs where dist < 2r
        overlap = (dist < 2 * r) & (dist > 1e-12)
        # Force magnitude proportional to overlap
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        # Normalize directions
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)  # (n, 2)
        
        # 2. Wall Constraints
        # Push away from walls if pts[i,k] + r > HALF or pts[i,k] - r < -HALF
        wall_force = np.zeros_like(pts)
        for k in range(2):
            # Right/Top wall
            too_right = pts[:, k] > HALF - r
            wall_force[too_right, k] = -(pts[too_right, k] - (HALF - r))
            # Left/Bottom wall
            too_left = pts[:, k] < -HALF + r
            wall_force[too_left, k] = (pts[too_left, k] - (-HALF + r))
            
        grad += wall_force
        
        # Update positions
        new_pts = pts + step * grad
        
        # Project back to valid bounds (center must be in [-HALF+r, HALF-r])
        new_pts = np.clip(new_pts, -HALF + r, HALF - r)
        
        # Check convergence / success
        # Recompute distances for new_pts
        diff_new = new_pts[:, None, :] - new_pts[None, :, :]
        dist_new = np.sqrt(np.sum(diff_new * diff_new, axis=2))
        np.fill_diagonal(dist_new, np.inf)
        min_pair_new = np.min(dist_new)
        min_wall_new = np.min(HALF - np.abs(new_pts))
        
        if min_pair_new >= 2 * r - 1e-10 and min_wall_new >= r - 1e-10:
            return True, new_pts
            
        # Adaptive step size reduction if we are oscillating or not improving
        # Simple heuristic: reduce step if min_pair_new is not significantly larger than min_pair
        # For simplicity, we just keep a constant step but maybe reduce it slightly over time
        # or use a fixed small step if we are close.
        # Let's try a simple decay
        if it % 50 == 0:
             step *= 0.9
             
        pts = new_pts
        
    # Final check
    dist_final = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist_final, np.inf)
    min_pair_final = np.min(dist_final)
    min_wall_final = np.min(HALF - np.abs(pts))
    success = (min_pair_final >= 2 * r - 1e-10) and (min_wall_final >= r - 1e-10)
    return success, pts

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # Initial best configuration
    # Start with a reasonable radius estimate, slightly lower than best known
    r_low = 0.06
    r_high = 0.08
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    # We want to find the max r such that _feasible returns True
    # We will run multiple iterations of binary search
    
    for it in range(20):  # ~1e-6 precision
        if time.time() > deadline:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try multiple seeds to avoid local minima
        seeds = [_hex_seed(r_mid), _grid_seed(r_mid)]
        
        # Add a perturbed seed
        perturbed = _hex_seed(r_mid) + np.random.randn(N, 2) * 0.01
        perturbed = np.clip(perturbed, -HALF + r_mid, HALF - r_mid)
        seeds.append(perturbed)
        
        success = False
        trial_pts = None
        
        for seed in seeds:
            if time.time() > deadline:
                break
            ok, pts = _feasible(seed, r_mid, max_iter=1000, deadline=deadline)
            if ok:
                success = True
                trial_pts = pts
                break
        
        if success:
            r_low = r_mid
            if trial_pts is not None:
                # Update best if this configuration is actually better (it should be, but check)
                r_new = _score(trial_pts)
                if r_new > best_r:
                    best_r = r_new
                    best_pts = trial_pts.copy()
        else:
            r_high = r_mid
            
        # If we've converged closely, stop
        if r_high - r_low < 1e-8:
            break
            
    # Final refinement phase
    # Try to push the best found configuration slightly further
    if time.time() < deadline - 10:
        current_r = _score(best_pts)
        # Try to increase radius by a small epsilon and see if we can relax
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            target_r = current_r + 0.0005
            ok, pts = _feasible(best_pts, target_r, max_iter=1000, deadline=deadline)
            if ok:
                new_r = _score(pts)
                if new_r > current_r:
                    best_pts = pts
                    current_r = new_r
                else:
                    break
            else:
                break
                
    # Ensure valid output
    best_pts = np.clip(best_pts, -0.499999999999, 0.499999999999)
    
    if not np.all(np.isfinite(best_points)) or best_pts.shape != (N, 2):
        return _hex_seed(0.07)
        
    return best_pts

# EVOLVE-BLOCK-START
# (The above construct function is the entry point)
# EVOLVE-BLOCK-END
