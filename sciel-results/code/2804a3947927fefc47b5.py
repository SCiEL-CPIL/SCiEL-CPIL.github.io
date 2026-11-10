import numpy as np
import time

N = 100

def _compute_radius(c):
    """
    Compute the largest common radius for given centers.
    c: (N, 3) array of centers.
    Returns: min(min_pair_dist/2, min_wall_dist)
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    # diff[i, j, k] = c[i, k] - c[j, k]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5] in each dimension
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _optimize(c_init, time_limit):
    """
    Vectorized Force-Directed Method with Simulated Annealing.
    c_init: Initial centers (N, 3)
    time_limit: Seconds to spend on this optimization attempt.
    Returns: Optimized centers, best radius found.
    """
    c = c_init.copy()
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    
    # Parameters
    # Initial step size should be small enough to not cause huge overlaps but large enough to move.
    # Typical radius ~0.11, so distances ~0.11. Step 0.01 is reasonable.
    step_size = 0.01
    # Temperature for noise
    temperature = 0.01
    # Decay rates
    step_decay = 0.995
    temp_decay = 0.998
    
    # Precompute indices for pairwise interactions to avoid recomputing every time?
    # With N=100, N^2=10000, which is small enough for direct broadcasting in NumPy.
    
    while time.time() - start_time < time_limit:
        
        # 1. Compute Forces
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-8, dists, 1e-8)
        
        # Force magnitude: 1/d^2 or similar. 
        # We want to push apart. 
        # F_ij = diff_ij / dists_ij^2
        # This creates a strong repulsion at close range.
        inv_dists2 = 1.0 / (dists_safe * dists_safe)
        inv_dists2[np.diag_indices(n)] = 0.0
        
        # Vectorized force calculation:
        # forces[i] = sum_j (c[i] - c[j]) / |c[i] - c[j]|^2
        # This is equivalent to: sum_j (diff[i,j] * inv_dists2[i,j])
        forces = np.sum(diff * inv_dists2[:, :, np.newaxis], axis=1)
        
        # 2. Wall Constraints / Repulsion
        # Push away from walls if too close.
        # Wall distance: 0.5 - |c|
        # If dist < threshold, add force towards center.
        wall_dist = 0.5 - np.abs(c)
        threshold = 0.1
        # Only apply force if close to wall
        mask = wall_dist < threshold
        # Force direction: -sign(c) (towards center)
        # Magnitude: proportional to how close we are
        wall_force = np.where(mask, (threshold - wall_dist) * np.sign(c) * 10.0, 0.0)
        forces += wall_force
        
        # 3. Normalize Forces (optional, but helps stability)
        # Limit max force per particle to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 1.0
        forces = np.where(force_norms > max_force, forces / force_norms * max_force, forces)
        
        # 4. Add Noise (Annealing)
        noise = np.random.randn(n, 3) * temperature
        total_move = step_size * forces + noise
        
        # 5. Update Positions
        c_new = c + total_move
        
        # 6. Clip to Cube (Hard Constraint)
        # Spheres must be inside. Centers must be in [-0.5+r, 0.5-r].
        # But we don't know r yet. We can just clip to [-0.5, 0.5] and let the radius calculation handle the "inside" part.
        # However, if a center goes outside, the radius becomes negative/invalid.
        # It's better to clip to [-0.5, 0.5] to keep them in the box, and the radius calc will penalize.
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # 7. Evaluate
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            # Reduce step size and temperature on success (cooling)
            step_size *= 0.99
            temperature *= 0.99
        else:
            # If it didn't improve, we might still accept it if it's "close" and we are hot?
            # Standard hill climbing: reject. 
            # To escape local minima, we rely on the noise. 
            # If we reject, we stay at c.
            # But we should still decay parameters to explore less wildly over time?
            # Or keep temperature constant to keep exploring?
            # Let's decay slowly even on failure to eventually settle.
            step_size *= 0.999
            temperature *= 0.999
            
            # If step size is too small, we might be stuck.
            # If we haven't improved for a long time, maybe increase noise?
            # Simple approach: just decay.
            
            # Check for stagnation: if best_r hasn't changed in many steps, boost noise?
            # For simplicity, we just let the decay handle it.
            
            # If parameters are too small, break early
            if step_size < 1e-6 and temperature < 1e-6:
                break
                
    return best_c, best_r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    """
    total_time_limit = 115.0  # Safety margin from 120s
    start_time = time.time()
    
    best_c = None
    best_r = -1.0
    
    # Strategy 1: Grid initialization
    # 100 spheres. 4x5x5 grid? 4*5*5 = 100.
    # Spacing: 1/4 = 0.25 in x, 1/5 = 0.2 in y, 1/5 = 0.2 in z.
    # This is a very regular start.
    
    # Generate grid points
    nx, ny, nz = 4, 5, 5
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid_points = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    c_grid = grid_points[:N].copy()
    
    # Run optimization on grid start
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_grid, t_left - 1.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Strategy 2: Random initialization
    # Random points in the cube
    c_rand = np.random.uniform(-0.4, 0.4, (N, 3))
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_rand, t_left - 1.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 3: Perturbed Grid
    # Add random noise to the grid to break symmetry
    c_perturbed = c_grid + np.random.randn(N, 3) * 0.05
    c_perturbed = np.clip(c_perturbed, -0.5, 0.5)
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_perturbed, t_left - 1.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 4: Second Random
    c_rand2 = np.random.uniform(-0.4, 0.4, (N, 3))
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_rand2, t_left - 1.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    # If we still have time, do a final polish on the best solution
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 2.0 and best_c is not None:
        c_opt, r_opt = _optimize(best_c, t_left - 0.5)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    if best_c is None:
        # Fallback: return grid
        return c_grid
        
    # Final safety check: ensure all points are strictly inside
    # The radius calculation ensures they are "inside" in terms of distance to wall > 0?
    # No, if a point is at 0.5, wall dist is 0, radius is 0.
    # We want to return the centers.
    return best_c

# EVOLVE-BLOCK-END
