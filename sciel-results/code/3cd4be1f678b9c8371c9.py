import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 50
HALF = 0.5
DEADLINE = 115.0  # Safety margin for 120s limit

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = np.min(np.abs(centers), axis=1)
    min_wall_dist = np.min(HALF - wall_dists)
    
    r = min(min_dist, min_wall_dist)
    return max(0.0, r)

def _hex_init():
    """
    Generate a high-quality hexagonal initialization.
    We generate a hex grid slightly larger than needed, then select the best N points
    or scale to fit.
    """
    # Target radius approx 0.0714
    # Hex grid spacing: dx = 2r, dy = r*sqrt(3)
    # Let's generate a grid of points and then optimize.
    
    # Create a dense hex grid
    r_guess = 0.075
    dx = 2 * r_guess
    dy = r_guess * np.sqrt(3)
    
    points = []
    # Cover a range larger than the square
    range_val = 1.0
    nx = int(range_val / dx) + 2
    ny = int(range_val / dy) + 2
    
    for i in range(-ny, ny):
        for j in range(-nx, nx):
            x = j * dx
            y = i * dy
            if i % 2 != 0:
                x += dx / 2
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the grid
    points -= np.mean(points, axis=0)
    
    # Select the N points closest to the center? 
    # Or select a subset that maximizes min distance?
    # Simple heuristic: Sort by distance to origin, take first N? 
    # This creates a "blob" which is good for packing.
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:N]
    points = points[indices]
    
    # Scale to fit inside the square with a small margin
    # We want max(|x|, |y|) + r <= 0.5
    # Current max extent
    max_ext = np.max(np.abs(points))
    # We don't know the final r yet, but we can scale so it fits comfortably.
    # If we scale too much, we lose potential.
    # Let's just return the points. The optimizer will handle the rest.
    # But they must be inside [-0.5, 0.5].
    # If max_ext > 0.5 - r_guess, we should scale down.
    target_max = HALF - r_guess
    if max_ext > target_max:
        scale = target_max / max_ext
        points *= scale
        
    return points

def _penalty_objective(centers_flat, r_target):
    """
    Objective for a fixed target radius r_target.
    Minimize the sum of squared violations of the non-overlapping and boundary constraints.
    """
    centers = centers_flat.reshape(N, 2)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We only care about pairs where dist < 2*r_target
    # Violation = (2*r_target - dist)^2
    # To make it smooth and only penalize violations:
    # v = np.maximum(0, 2*r_target - dists)
    # penalty = np.sum(v**2)
    
    # However, computing this for all 1225 pairs every time is fast in numpy.
    v_pairs = np.maximum(0, 2 * r_target - dists)
    penalty_pairs = np.sum(v_pairs ** 2)
    
    # Wall violations
    # wall_dist = min(0.5 - |x|, 0.5 - |y|)
    # v_wall = np.maximum(0, r_target - wall_dist)
    
    # Vectorized wall distance
    # x: [N], y: [N]
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to 4 walls
    d_left = HALF + x
    d_right = HALF - x
    d_bottom = HALF + y
    d_top = HALF - y
    
    # Min distance to any wall
    d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    v_walls = np.maximum(0, r_target - d_wall)
    penalty_walls = np.sum(v_walls ** 2)
    
    return penalty_pairs + penalty_walls

def _is_feasible(centers, r_target, tol=1e-6):
    """Check if centers are feasible for r_target."""
    return _penalty_objective(centers.flatten(), r_target) < tol

def _optimize_for_radius(r_target, x0, max_iter=500):
    """
    Optimize positions to satisfy radius r_target.
    Returns (success, centers).
    """
    bounds = [(-HALF, HALF)] * (2 * N)
    
    # Use L-BFGS-B as it handles bounds well and is fast for this smooth penalty.
    try:
        res = minimize(
            lambda c: _penalty_objective(c, r_target),
            x0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        new_centers = res.x.reshape(N, 2)
        # Clip to be safe
        new_centers = np.clip(new_centers, -HALF, HALF)
        
        # Check feasibility
        penalty = _penalty_objective(new_centers.flatten(), r_target)
        if penalty < 1e-6:
            return True, new_centers
        else:
            return False, new_centers
    except Exception:
        return False, x0

def construct():
    start_time = time.time()
    
    # 1. Initialization
    centers = _hex_init()
    current_r = _compute_radius(centers)
    
    best_centers = centers.copy()
    best_r = current_r
    
    # 2. Binary Search on Radius with Optimization
    # We know r is between 0.07 and 0.08 approx.
    # Let's do a coarse search first, then refine.
    
    # Coarse Search
    r_low = 0.06
    r_high = 0.08
    
    # Ensure initial guess is feasible for r_low
    # If not, reduce r_low
    while not _is_feasible(centers, r_low):
        r_low *= 0.9
        if r_low < 0.01: break
        
    # Try to find a feasible r in [r_low, r_high]
    # We will perform a binary search.
    # For each candidate r, we try to optimize the positions.
    # If successful, r_low = r. If failed, r_high = r.
    
    # Number of binary search steps: ~20 steps gives precision 0.02 / 2^20 ~ 2e-8
    # But each step involves an optimization run.
    # Optimization run takes ~0.1-1s.
    # 20 steps * 1s = 20s. This is well within budget.
    # We can afford more steps or restarts.
    
    # Let's do 25 steps of binary search.
    num_steps = 25
    
    for i in range(num_steps):
        if time.time() > DEADLINE:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to optimize for r_mid
        # Use best_centers as starting point
        success, new_centers = _optimize_for_radius(r_mid, best_centers)
        
        if success:
            r_low = r_mid
            best_centers = new_centers
            best_r = r_mid
        else:
            r_high = r_mid
            
        # Update best_r if we found a better valid radius during optimization
        # (Sometimes the optimizer finds a valid config for r_mid even if penalty > tol,
        # but strictly speaking, if penalty > tol, it's not valid for r_mid.
        # However, _compute_radius gives the actual radius.
        # Let's compute the actual radius of new_centers and update best if better.
        actual_r = _compute_radius(new_centers)
        if actual_r > best_r:
            best_r = actual_r
            best_centers = new_centers
            
    # 3. Final Polish
    # After binary search, we have a good best_centers.
    # We can try to squeeze a bit more by optimizing for best_r + epsilon.
    # Or just run a few local optimizations on best_centers to maximize radius directly.
    
    # Direct Maximization Polish
    # Use L-BFGS-B to maximize -min_dist directly? 
    # Or use the penalty method with r = best_r + small_delta.
    
    if time.time() < DEADLINE - 10:
        # Try to improve by small steps
        current_r = best_r
        for _ in range(5):
            if time.time() > DEADLINE - 5:
                break
            next_r = current_r * 1.001
            success, new_centers = _optimize_for_radius(next_r, best_centers, max_iter=200)
            if success:
                current_r = next_r
                best_centers = new_centers
                best_r = current_r
            else:
                break
                
    # Final validation
    best_centers = np.clip(best_centers, -HALF, HALF)
    
    return best_centers
