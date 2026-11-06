"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist

N = 200

def _compute_radius(pos):
    """Compute the maximum valid radius for a given set of centers."""
    # Boundary constraint
    min_boundary = 0.5 - np.max(np.abs(pos))
    # Pairwise constraint
    if len(pos) < 2:
        return min_boundary
    dists = pdist(pos)
    min_dist = np.min(dists)
    min_pair = min_dist / 2.0
    return min(min_boundary, min_pair)

def _initial_hexagonal(n_points):
    """Create a hexagonal lattice initial placement."""
    # Hexagonal packing density is pi / (2 * sqrt(3)) ≈ 0.9069
    # For N points in area 1, approximate spacing d:
    # N * (sqrt(3)/2 * d^2) ≈ 1  => d ≈ sqrt(2 / (N * sqrt(3)))
    d = np.sqrt(2.0 / (n_points * np.sqrt(3.0)))
    
    # Number of rows
    # Each row has approx 1/d points.
    # Row height is sqrt(3)/2 * d.
    # Total height ≈ rows * sqrt(3)/2 * d ≈ 1
    # rows ≈ 2 / (sqrt(3) * d)
    rows = int(np.ceil(2.0 / (np.sqrt(3.0) * d)))
    
    points = []
    for i in range(rows):
        y = -0.5 + (i + 0.5) * np.sqrt(3.0) / 2.0 * d
        if y > 0.5:
            break
        # Offset for even rows
        offset = (np.sqrt(3.0) / 2.0 * d) / 2.0 if i % 2 == 1 else 0.0
        x_start = -0.5 + d / 2.0 + offset
        x = x_start
        while x < 0.5 - d / 2.0 + 1e-9:
            points.append([x, y])
            x += d
        
        if len(points) >= n_points:
            break
            
    points = np.array(points[:n_points])
    
    # If we don't have enough points, add random ones or extend grid
    if len(points) < n_points:
        # Fallback to grid
        cols = int(np.ceil(np.sqrt(n_points)))
        rows_g = int(np.ceil(n_points / cols))
        xs = np.linspace(-0.5 + 0.5/cols, 0.5 - 0.5/cols, cols)
        ys = np.linspace(-0.5 + 0.5/rows_g, 0.5 - 0.5/rows_g, rows_g)
        grid = []
        for y in ys:
            for x in xs:
                grid.append([x, y])
                if len(grid) == n_points:
                    break
            if len(grid) == n_points:
                break
        points = np.array(grid)
        
    # Add small perturbation
    np.random.seed(42)
    points += np.random.uniform(-0.001, 0.001, points.shape)
    
    # Clip to square
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    return points

def _repulsion_force(pos, r_target):
    """
    Compute repulsive forces for circles overlapping with target radius r_target.
    Also includes wall forces.
    Returns gradient of a penalty function (to be minimized).
    """
    n = len(pos)
    forces = np.zeros_like(pos)
    
    # Pairwise repulsion
    # Use a soft repulsion: if dist < 2*r_target, push apart.
    # Force magnitude proportional to (2*r_target - dist)
    
    # Vectorized calculation is faster but memory heavy for N=200 (40k pairs).
    # 40k is fine.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Only consider pairs that are too close
    # We want to minimize sum (max(0, 2r - dist))^2
    # Gradient wrt pos_i: -2 * max(0, 2r - dist) * (pos_i - pos_j) / dist
    # Wait, let's just compute the force directly.
    
    # For each pair (i, j) with dist < 2*r_target:
    # Force on i is in direction (pos_i - pos_j) / dist * magnitude
    # Magnitude = (2*r_target - dist)
    
    # To avoid division by zero, add epsilon
    dists_safe = np.where(dists > 1e-10, dists, 1e-10)
    
    # Compute violations
    violations = np.maximum(0, 2 * r_target - dists)
    
    # Force direction
    # diff is (i, j) -> pos_i - pos_j
    # We want to push i away from j, so force on i is along +diff
    # Force on j is along -diff
    
    # Weighted diff
    # weight = violations / dists
    weights = violations / dists_safe
    np.fill_diagonal(weights, 0)
    
    # Sum of forces on each particle
    # F_i = sum_j weights[i,j] * diff[i,j]
    # Note: diff[i,j] = pos_i - pos_j
    forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    # Wall forces
    # If pos_i > 0.5 - r_target, push left (negative x)
    # If pos_i < -0.5 + r_target, push right (positive x)
    # Same for y
    
    limit = 0.5 - r_target
    
    # X boundaries
    # Right wall: x > limit
    forces[:, 0] -= np.maximum(0, pos[:, 0] - limit)
    # Left wall: x < -limit
    forces[:, 0] += np.maximum(0, -limit - pos[:, 0])
    
    # Y boundaries
    # Top wall: y > limit
    forces[:, 1] -= np.maximum(0, pos[:, 1] - limit)
    # Bottom wall: y < -limit
    forces[:, 1] += np.maximum(0, -limit - pos[:, 1])
    
    return forces

def _optimize_repulsion(pos, r_target, time_limit, step_size=0.001):
    """
    Iteratively adjust positions to resolve overlaps for a given target radius.
    Returns the updated positions and success flag.
    """
    start_time = time.time()
    current_pos = pos.copy()
    
    # Adaptive step size
    lr = step_size
    prev_violation = np.inf
    
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        forces = _repulsion_force(current_pos, r_target)
        
        # Check convergence
        # Max force magnitude
        max_force = np.max(np.linalg.norm(forces, axis=1))
        
        if max_force < 1e-8:
            break
            
        # Update positions
        current_pos += lr * forces
        
        # Keep within square (hard constraint)
        current_pos[:, 0] = np.clip(current_pos[:, 0], -0.5, 0.5)
        current_pos[:, 1] = np.clip(current_pos[:, 1], -0.5, 0.5)
        
        # Check for divergence or oscillation
        # Compute current violation
        dists = pdist(current_pos)
        min_dist = np.min(dists)
        min_boundary = 0.5 - np.max(np.abs(current_pos))
        current_violation = max(2 * r_target - min_dist, 0) + max(min_boundary - r_target, 0)
        
        if current_violation > prev_violation * 1.5:
            # Step was too large
            lr *= 0.5
            current_pos = pos.copy() # Revert? No, just reduce lr and try again from current?
            # Actually, reverting is safer
            # But we already updated. Let's just reduce lr.
        else:
            # If improvement is small, we might be stuck.
            if prev_violation - current_violation < 1e-10:
                lr *= 0.9
            else:
                lr = min(lr * 1.05, 0.01) # Slowly increase lr if stable
            prev_violation = current_violation
            
    return current_pos

def construct():
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin
    
    # Initial placement
    pos = _initial_hexagonal(N)
    
    # Initial radius estimate
    # Hexagonal packing radius approx
    # d = sqrt(2 / (N * sqrt(3)))
    # r = d / 2
    r_est = np.sqrt(2.0 / (N * np.sqrt(3.0))) / 2.0
    
    # We know optimal is around 0.0366
    # Let's start a bit lower
    r_current = r_est * 0.9
    
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Bisection on radius
    r_low = 0.030
    r_high = 0.038
    
    # First, ensure we have a valid starting point for r_low
    # Optimize for r_low
    pos = _optimize_repulsion(pos, r_low, time_limit=deadline - (time.time() - start_time) - 5)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Binary search for max radius
    # We will try to find the largest r such that we can pack N circles.
    # Since the problem is non-convex, "can pack" is hard.
    # Instead, we iteratively increase r and try to resolve overlaps.
    # If we succeed, keep it. If we fail (stuck with overlaps), we might be stuck.
    # A more robust approach:
    # 1. Start with a valid configuration for some r0.
    # 2. Increase r slightly.
    # 3. Run repulsion optimization.
    # 4. If the resulting configuration has min_radius >= r_new, accept.
    # 5. Otherwise, revert and stop or try smaller increment.
    
    r_target = best_r
    improvement_found = True
    
    while improvement_found and time.time() - start_time < deadline - 10:
        # Try to increase radius
        # Step size adaptive
        delta_r = max(0.0001, (r_high - r_low) * 0.1)
        r_new = r_target + delta_r
        
        # Don't go too high
        if r_new > 0.04:
            break
            
        # Optimize for new radius
        # Use current best_pos as start
        pos_new = _optimize_repulsion(best_pos, r_new, time_limit=10.0)
        
        # Check if valid
        r_achieved = _compute_radius(pos_new)
        
        if r_achieved >= r_new - 1e-9:
            # Success
            best_pos = pos_new
            best_r = r_achieved
            r_target = r_new
            r_low = r_new
            # Increase delta_r slightly? Or keep same?
            # Let's keep delta_r constant or slightly increase
            delta_r = min(delta_r * 1.5, 0.001)
        else:
            # Failure
            # Try smaller step
            delta_r *= 0.5
            if delta_r < 1e-6:
                improvement_found = False
            # Don't update best_pos
            # Maybe try a different initial perturbation?
            # For now, just stop increasing.
            
    # Final polish: try to squeeze out a bit more using smaller steps
    # This is essentially a local search.
    # We can also try random restarts of the repulsion optimization with different initial perturbations.
    
    # Let's do a few random restarts if time permits
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 20:
        np.random.seed(43)
        for _ in range(5):
            if time.time() - start_time > deadline - 5:
                break
            # Perturb best_pos
            perturbed_pos = best_pos + np.random.uniform(-0.002, 0.002, best_pos.shape)
            perturbed_pos[:, 0] = np.clip(perturbed_pos[:, 0], -0.49, 0.49)
            perturbed_pos[:, 1] = np.clip(perturbed_pos[:, 1], -0.49, 0.49)
            
            # Try to optimize for a slightly higher radius
            r_try = best_r + 0.0001
            pos_new = _optimize_repulsion(perturbed_pos, r_try, time_limit=5.0)
            r_achieved = _compute_radius(pos_new)
            
            if r_achieved > best_r:
                best_pos = pos_new
                best_r = r_achieved
                
    # Final check and clip
    best_pos[:, 0] = np.clip(best_pos[:, 0], -0.5, 0.5)
    best_pos[:, 1] = np.clip(best_pos[:, 1], -0.5, 0.5)
    
    return best_pos

# EVOLVE-BLOCK-END
