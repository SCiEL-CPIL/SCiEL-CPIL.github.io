"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses a repulsion-based optimization with adaptive radius targeting.
    """
    # Time management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Helper function to generate initial hexagonal lattice points
    def get_hex_grid_points(n_points):
        # Estimate spacing
        # Area of unit circle is pi. Area per circle approx pi*r^2.
        # r ~ 0.05. Area ~ 0.0078.
        # Spacing ~ 2*r ~ 0.1.
        # Let's just generate a dense grid and pick the best n_points
        # based on distance from center.
        
        # Generate a grid of points
        # Max radius 1.
        # Spacing roughly 0.1
        step = 0.05
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        
        xs = np.arange(x_min, x_max + step, step)
        ys = np.arange(y_min, y_max + step, step)
        
        # Hexagonal offset
        points = []
        for i, x in enumerate(xs):
            offset = 0.0 if i % 2 == 0 else step * np.sqrt(3) / 2
            for y in ys:
                y_off = y + offset
                if x**2 + y_off**2 <= 1.0:
                    points.append((x, y_off))
        
        points = np.array(points)
        
        # Sort by distance from center to pick the most central ones?
        # Actually, for packing, we want a uniform distribution.
        # The hex grid is already uniform.
        # If we have more than N, we need to select N.
        # Selecting the N closest to the center is a good heuristic for initial packing.
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        return points[indices[:n_points]]

    # Helper to calculate the valid radius for a set of points
    def calc_radius(points):
        # points: (N, 2)
        # Pairwise distances
        # For N=300, 300x300 matrix is fine (90k entries)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        norm_points = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - norm_points)
        
        r = min(min_pair_dist / 2.0, min_bound_dist)
        return r

    # Helper to run repulsion optimization for a target radius
    def optimize_for_radius(points, target_r, max_iters, lr_initial, lr_decay):
        """
        Attempts to arrange points to fit circles of radius target_r.
        Returns success (bool) and the optimized points.
        """
        points = points.copy()
        lr = lr_initial
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Calculate forces
            # Pairwise
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-10, dists, 1e-10)
            
            # Overlap amount: max(0, 2*r - dist)
            overlaps = np.maximum(0, 2.0 * target_r - dists)
            
            # Force magnitude: proportional to overlap
            # Direction: along the vector
            # F_ij on point i due to j: -overlaps_ij * diff_ij / dists_ij
            # We sum forces on each point.
            
            # Normalize diff to get unit vectors
            unit_diff = diff / dists_safe[:, :, np.newaxis]
            
            # Force contribution from pairs
            # F_i = sum_j (overlaps_ij * unit_diff_ij)
            # Note: diff_ij = c_i - c_j. If i is too close to j, we want to push i away from j.
            # So force on i should be in direction (c_i - c_j).
            # overlaps is symmetric.
            forces = np.sum(overlaps[:, :, np.newaxis] * unit_diff, axis=1)
            
            # Boundary forces
            # If point is too close to boundary (dist > 1 - r), push in.
            norms = np.linalg.norm(points, axis=1)
            boundary_violations = np.maximum(0, norms - (1.0 - target_r))
            # Direction: towards center = -points / norms
            unit_inward = -points / np.maximum(norms[:, np.newaxis], 1e-10)
            boundary_forces = boundary_violations[:, np.newaxis] * unit_inward
            
            # Total force
            total_forces = forces + boundary_forces
            
            # Update
            points += lr * total_forces
            
            # Decay learning rate
            lr *= lr_decay
            
            # Optional: Check if we are stuck or converged
            # We can check max overlap
            # max_overlap = np.max(overlaps)
            # if max_overlap < 1e-6:
            #     return True, points
            
        # Final check
        # We don't strictly need success/fail, we just return the points.
        # The caller will check if the radius improved.
        return points

    # Initial configuration
    # Use hex grid
    points = get_hex_grid_points(N)
    
    # Shuffle slightly to break symmetry? Hex grid is symmetric.
    # Add small random noise
    noise = np.random.normal(0, 0.001, size=points.shape)
    points += noise
    
    # Calculate initial radius
    current_r = calc_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Optimization loop
    # We try to increase the radius gradually.
    # Start with current_r and try to increase it.
    
    # To be robust, we might do multiple restarts.
    # Given time constraints, we do one main run with internal restarts if stuck.
    
    target_r = current_r * 1.001
    lr_initial = 0.05
    lr_decay = 0.999
    max_iters_per_step = 1000
    
    # Main loop
    while time.time() - start_time < time_limit:
        # Run optimization for target_r
        optimized_points = optimize_for_radius(points, target_r, max_iters_per_step, lr_initial, lr_decay)
        
        # Calculate new radius
        new_r = calc_radius(optimized_points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = optimized_points.copy()
            points = optimized_points.copy()
            # Increase target radius slightly
            target_r = new_r * 1.0005
            # Reset learning rate? Or keep it low?
            # Let's reset to a moderate value to allow movement
            lr_initial = 0.05
        else:
            # No improvement. 
            # Maybe we are stuck.
            # Perturb the points and try again.
            # Or reduce target_r? No, we want to maximize.
            # If we can't fit target_r, we are stuck in a local min for that radius.
            # Let's try a random restart with a slightly better initial guess or just perturb.
            
            # Perturb current points
            points = best_points + np.random.normal(0, 0.01, size=best_points.shape)
            # Ensure points stay inside
            norms = np.linalg.norm(points, axis=1)
            # If any point is outside, pull it back
            mask = norms > (1.0 - best_r)
            if np.any(mask):
                points[mask] = best_points[mask] # Fallback to best for those
            
            # Try a slightly smaller target to see if we can escape?
            # Or just try the same target with perturbed start.
            target_r = best_r * 1.0005
            
            # If we've been stuck for a long time, do a full restart with hex grid + more noise
            # We can track iterations without improvement.
            # For simplicity, we just perturb.
            
            # If perturbation doesn't work, we might need to change the structure.
            # Let's limit the number of "stuck" iterations by restarting the optimization parameters.
            lr_initial = 0.1 # Higher learning rate for escape
            lr_decay = 0.995

    return best_points

# EVOLVE-BLOCK-END
