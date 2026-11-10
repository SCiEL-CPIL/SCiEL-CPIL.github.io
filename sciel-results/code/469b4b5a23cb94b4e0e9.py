"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(centres):
    """
    Compute the maximum radius for the given centres.
    r = min( min_{i<j} dist_ij / 2, min_i wall_dist_i )
    """
    if centres.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Wall distances: distance from centre to nearest wall (which is 0.5 - |coord|)
    # Since cube is [-0.5, 0.5], distance to wall is 0.5 - max(|x|, |y|, |z|) ? 
    # No, distance to the boundary of the box.
    # The distance from a point (x,y,z) to the boundary of [-0.5, 0.5]^3 is min(0.5-|x|, 0.5-|y|, 0.5-|z|).
    abs_centres = np.abs(centres)
    wall_dists = 0.5 - np.max(abs_centres, axis=1)
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist <= 0:
        return 0.0

    # Pairwise distances
    # Using broadcasting for 50x50 is fine
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _relax(centres, target_r, time_limit=10.0):
    """
    Relax the configuration to satisfy min_dist >= 2*target_r and wall_dist >= target_r.
    Uses a simple velocity Verlet or Euler integration with forces.
    """
    # Copy to avoid modifying input
    c = centres.copy()
    
    # Initial velocities
    v = np.zeros_like(c)
    
    dt = 0.01
    # Damping
    damping = 0.95
    
    start_time = time.time()
    steps = 0
    
    # We want to maximize the minimum distance, but we are targeting a specific r.
    # Force model:
    # 1. Pair repulsion if dist < 2*target_r
    # 2. Wall repulsion if wall_dist < target_r
    
    while time.time() - start_time < time_limit:
        forces = np.zeros_like(c)
        
        # Pairwise forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs closer than 2*target_r + small_epsilon
        # Actually, to be smooth, we can use a potential like 1/d^2 or similar, but cut off.
        # Let's use a simple linear spring-like force for overlaps.
        # Force magnitude proportional to overlap depth.
        
        # Efficient vectorized force calculation for N=50 is easy.
        # F_ij = k * (2*r - d) * (d_ij / d)  if d < 2*r
        
        mask = dists < 2 * target_r
        # We need to sum forces on i from all j.
        # diff is (i, j, dim).
        # Unit vector from i to j is diff / dist.
        # Force on i from j is in direction of (j - i) = -diff.
        # Wait, diff = c_i - c_j. So c_j - c_i = -diff.
        # Force on i should push i away from j. So direction is (c_i - c_j) = diff.
        # Magnitude k * (2*r - d).
        
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        
        # Force vector from j to i:
        # F = k * (2*r - d) * (diff / d)
        # Only where mask is True.
        
        # Create a matrix of force magnitudes
        mag = np.zeros_like(dists)
        mag[mask] = (2 * target_r - dists[mask])
        
        # Normalize diff
        # diff is (N, N, 3)
        # safe_dists is (N, N)
        # We want diff / dist for each component.
        # diff / safe_dists[..., np.newaxis]
        
        unit_diff = diff / safe_dists[..., np.newaxis]
        
        # Force on i from j is mag[i,j] * unit_diff[i,j]
        # Sum over j (axis 1)
        pair_forces = np.sum(mag[..., np.newaxis] * unit_diff, axis=1)
        
        forces += pair_forces
        
        # Wall forces
        # For each dimension, if c < -0.5 + r, push right (+). If c > 0.5 - r, push left (-).
        # Overlap depth:
        # Left wall: (-0.5 + r) - c. If > 0, push +.
        # Right wall: c - (0.5 - r). If > 0, push -.
        
        for dim in range(3):
            left_wall = -0.5 + target_r
            right_wall = 0.5 - target_r
            
            # Overlap with left
            left_overlap = left_wall - c[:, dim]
            # If left_overlap > 0, force is +1 (scaled)
            f_left = np.where(left_overlap > 0, left_overlap, 0.0)
            
            # Overlap with right
            right_overlap = c[:, dim] - right_wall
            # If right_overlap > 0, force is -1 (scaled)
            f_right = np.where(right_overlap > 0, right_overlap, 0.0)
            
            forces[:, dim] += f_left - f_right
        
        # Update velocities
        v += forces * dt
        v *= damping
        
        # Update positions
        c += v * dt
        
        # Hard constraint: keep inside box [-0.5, 0.5]
        # If a sphere goes out, clamp it. This prevents explosion.
        c = np.clip(c, -0.5, 0.5)
        
        # Check convergence
        # If max force is small, we might be at a local minimum for this target_r.
        # But we are just trying to find *a* valid config for this r.
        # Actually, we want to know if we CAN fit r.
        # If after relaxation, the min_dist >= 2r and wall_dist >= r, then r is feasible.
        
        # Early exit if feasible?
        # Calculating score every step is expensive. Do it every 50 steps.
        steps += 1
        if steps % 50 == 0:
            # Check if we satisfied the target
            # Simple check: min wall dist and min pair dist
            abs_c = np.abs(c)
            w_dists = 0.5 - np.max(abs_c, axis=1)
            if np.min(w_dists) < target_r:
                continue
            
            # Pairwise
            d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            ds = np.linalg.norm(d, axis=2)
            np.fill_diagonal(ds, np.inf)
            if np.min(ds) >= 2 * target_r:
                # Feasible!
                return c, True
                
    # Final check
    abs_c = np.abs(c)
    w_dists = 0.5 - np.max(abs_c, axis=1)
    if np.min(w_dists) < target_r:
        return c, False
        
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    ds = np.linalg.norm(d, axis=2)
    np.fill_diagonal(ds, np.inf)
    if np.min(ds) >= 2 * target_r:
        return c, True
    else:
        return c, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centres = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    init_centres += np.random.normal(0, 0.01, init_centres.shape)
    init_centres = np.clip(init_centres, -0.49, 0.49)
    
    # Binary search on radius
    # Lower bound: 0
    # Upper bound: theoretical max for 50 spheres. 
    # Volume argument: 50 * (4/3)pi r^3 <= 1 => r <= (3/(20pi))^(1/3) ~ 0.24
    # Pfoertner is 0.136. Let's start high and go down, or start low and go up.
    # Going up is safer for validity.
    
    low = 0.0
    high = 0.15 # Start above known best
    
    # First, ensure we have a valid config for some small r to start with.
    # The grid config has a certain radius.
    current_best_centres = init_centres
    current_best_score = _compute_score(current_best_centres)
    
    # If initial score is 0, we have a problem. Jitter should fix it.
    if current_best_score <= 0:
         # Try to relax a very small radius
         c, ok = _relax(init_centres, 0.01, time_limit=5.0)
         if ok:
             current_best_centres = c
             current_best_score = 0.01
         else:
             # Fallback: return the grid, hoping it's valid (it usually is for small N, but N=50 grid is tight)
             # Actually, grid of 50 (4x4x4=64) has spacing 1/4=0.25. r=0.125.
             # So it should be valid.
             pass

    # Binary search
    # We will try to find the max r such that we can relax to a valid config.
    
    # To save time, we do a coarse search first, then fine.
    # Or just standard binary search with a time limit on each relaxation.
    
    iterations = 0
    max_iterations = 20 # ~ 20 * 5s = 100s. Safe.
    
    for _ in range(max_iterations):
        if time.time() > deadline - 20.0: # Leave 20s for final polish and return
            break
            
        mid = (low + high) / 2.0
        
        # Try to find a config for radius 'mid'
        # We can start from the previous best config or a fresh grid?
        # Starting from previous best config (which was valid for 'low') is better.
        # But if 'low' was 0, we start from grid.
        
        # Let's use the current best centres as the starting point for relaxation.
        # This helps continuity.
        
        relaxed_centres, feasible = _relax(current_best_centres, mid, time_limit=10.0)
        
        if feasible:
            # Update best
            score = _compute_score(relaxed_centres)
            if score > current_best_score:
                current_best_score = score
                current_best_centres = relaxed_centres
            low = mid
        else:
            high = mid
            
    # Final polish: Try to improve the current best config slightly
    # Use a very small relaxation with a target slightly higher than current score?
    # Or just return the best found.
    
    # One last check: ensure the returned centres are strictly valid.
    final_score = _compute_score(current_best_centres)
    if final_score <= 0:
        # Emergency fallback: return the grid
        return grid[:N]
        
    return current_best_centres

# EVOLVE-BLOCK-END
