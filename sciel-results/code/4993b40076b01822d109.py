"""Pack 50 equal spheres in a unit cube to maximize the common radius."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 50
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum feasible radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: distance from each center to the nearest face
    dist_to_boundary = np.min(CUBE_HALF - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return 0.5 * min(min_boundary, min_pair)

def _initial_placement():
    """Create an initial valid placement using a grid."""
    # Use a 4x4x4 grid = 64 positions, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _project_to_feasible(centers, r):
    """
    Project centers to be feasible for radius r.
    This is a simplified projection that just clamps to boundaries
    and does a few iterations of repulsion for overlaps.
    """
    centers = centers.copy()
    # Clamp to boundaries
    centers = np.clip(centers, -CUBE_HALF + r, CUBE_HALF - r)
    
    # Simple repulsion iterations
    for _ in range(10):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                min_dist = 2 * r
                if dist < min_dist and dist > 1e-12:
                    # Push apart
                    push = (min_dist - dist) / 2.0
                    direction = d / dist
                    centers[i] += push * direction
                    centers[j] -= push * direction
                    moved = True
            # Re-clamp
            centers[i] = np.clip(centers[i], -CUBE_HALF + r, CUBE_HALF - r)
    return centers

def _optimize_for_radius(r):
    """
    Try to find a feasible placement for radius r.
    Returns (success, centers).
    """
    # Start from a grid-like placement
    centers = _initial_placement()
    
    # Objective: minimize sum of squared violations
    def objective(x):
        c = x.reshape(N, 3)
        # Boundary violations
        boundary_viol = np.maximum(0, np.abs(c) - (CUBE_HALF - r))
        # Pairwise violations
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0, 2 * r - dist)
        return np.sum(boundary_viol ** 2) + np.sum(pair_viol ** 2)
    
    # Try gradient descent / Nelder-Mead
    best_centers = centers.copy()
    best_obj = objective(centers.flatten())
    
    # Use a simple iterative improvement
    for iteration in range(200):
        # Compute gradient numerically or analytically
        # For simplicity, use a small perturbation approach
        improved = False
        for i in range(N):
            for axis in range(3):
                # Try moving center i along axis
                step = 0.001
                for direction in [-1, 1]:
                    new_center = best_centers.copy()
                    new_center[i, axis] += direction * step
                    # Clamp
                    new_center = np.clip(new_center, -CUBE_HALF, CUBE_HALF)
                    obj = objective(new_center.flatten())
                    if obj < best_obj:
                        best_obj = obj
                        best_centers = new_center
                        improved = True
                        break
        if not improved:
            break
    
    # Check if feasible
    c = best_centers
    dist_to_boundary = np.min(CUBE_HALF - np.abs(c), axis=1)
    if np.min(dist_to_boundary) < r - 1e-9:
        return False, best_centers
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 2 * r - 1e-9:
        return False, best_centers
    
    return True, best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Start with a good initial placement
    centers = _initial_placement()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Bisection search on radius
    # Upper bound: theoretical max is when spheres fill the cube
    # For 50 spheres in unit cube, r < 0.5
    # Practical upper bound from known results ~0.136
    r_low = best_r
    r_high = 0.14  # Slightly above known best
    
    # First, try to improve from initial placement
    while time.time() < deadline:
        # Try to find feasible placement for mid radius
        r_mid = (r_low + r_high) / 2
        if r_mid <= best_r + 1e-12:
            # We've found a better radius
            r_low = r_mid
            continue
        
        success, centers_candidate = _optimize_for_radius(r_mid)
        
        if success:
            # Found feasible placement for r_mid
            actual_r = _compute_radius(centers_candidate)
            if actual_r > best_r:
                best_r = actual_r
                best_centers = centers_candidate.copy()
            r_low = r_mid
        else:
            r_high = r_mid
        
        # If we've converged enough
        if r_high - r_low < 1e-6:
            break
        
        # Check time
        if time.time() > deadline - 20:
            break
    
    # Final check and return
    final_r = _compute_radius(best_centers)
    if final_r < 1e-10:
        # Fallback to simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
