import numpy as np
import time

N = 50
DEADLINE = 145.0  # Leave 5s margin for safety

def _compute_radius(c):
    """
    Compute the radius for a given configuration c of shape (N, 3).
    Returns the minimum of (min_pair_dist / 2) and (min_wall_dist).
    """
    if c.shape != (N, 3):
        return 0.0
    
    # Wall distance: min_i min_k (0.5 - |c_ik|)
    # Equivalent to 0.5 - max_i max_k |c_ik|
    wall_dist = 0.5 - np.max(np.abs(c))
    
    # Pairwise distances
    # To save memory and time for N=50, we can compute the full matrix.
    # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=2))
    # np.fill_diagonal(dists, np.inf)
    # min_pair_dist = np.min(dists)
    
    # More efficient: compute squared distances, take min, then sqrt
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    
    return min(wall_dist, min_pair_dist / 2.0)

def _local_refine(c, start_time, max_time):
    """
    Custom local search to maximize the minimum distance.
    Uses a simple steepest ascent approach on the 'bottleneck' constraints.
    """
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # Parameters for the search
    step_size = 0.01
    min_step = 1e-6
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - start_time < max_time:
        # Compute distances
        diff = best_c[:, np.newaxis, :] - best_c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Wall distances
        wall_dists = 0.5 - np.max(np.abs(best_c), axis=1)
        
        current_r = min(np.min(wall_dists), np.min(dists) / 2.0)
        
        # Identify critical spheres
        # 1. Spheres violating or close to wall constraint
        # 2. Spheres involved in closest pair constraints
        
        # We want to move spheres to increase the minimum clearance.
        # Force vector approach:
        # For each pair (i, j) with dist < current_r * 2 (or close), push apart.
        # For each sphere i with wall_dist < current_r, push inward.
        
        forces = np.zeros_like(best_c)
        
        # Pairwise forces: push apart if too close
        # We only care about pairs that are "active" (close to the minimum)
        # Let's use a threshold slightly above the current minimum to create a gradient
        threshold = current_r * 2.0 + 1e-4
        
        # Mask for active pairs
        active_pairs = dists < threshold
        # Exclude diagonal
        active_pairs.fill_diagonal(False)
        
        # Sum up forces for each sphere
        # For each active pair (i, j), force on i is (c_i - c_j) / dist
        # Force on j is (c_j - c_i) / dist
        
        # Efficient way:
        # diff is c_i - c_j
        # force_vec = diff / dists[:, :, np.newaxis]
        # But we only want to sum where active_pairs is True.
        
        # To avoid division by zero for non-active or diagonal, we can use a masked array or careful broadcasting.
        # Since N=50, a simple loop might be faster than complex vectorization if we do it right, 
        # but vectorized is usually better.
        
        # Let's compute normalized diff
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_dists = np.where(active_pairs, dists, 1.0)
            norm_diff = diff / norm_dists[:, :, np.newaxis]
        
        # Zero out non-active pairs
        norm_diff = np.where(active_pairs[:, :, np.newaxis], norm_diff, 0.0)
        
        # Sum forces for each sphere i: sum over j of norm_diff[i, j]
        pair_forces = np.sum(norm_diff, axis=1)
        forces += pair_forces
        
        # Wall forces: push inward if close to wall
        # If c_ik is close to 0.5, force is -sign(c_ik)
        # If c_ik is close to -0.5, force is +1
        # We want to maximize min(0.5 - |c|).
        # Gradient of (0.5 - |c|) is -sign(c).
        # So force should be -sign(c) for spheres that are "active" (close to wall).
        
        wall_threshold = current_r + 1e-4
        active_walls = wall_dists < wall_threshold
        
        # Compute wall force for each sphere
        # If active, force = -sign(c)
        # If not active, force = 0
        # Note: if c is 0, sign is 0, which is fine (any direction is ok, but we can add small noise or just let pair forces handle it)
        
        wall_forces = np.where(active_walls[:, np.newaxis], -np.sign(best_c), 0.0)
        forces += wall_forces
        
        # Normalize forces to avoid large steps
        force_norms = np.linalg.norm(forces, axis=1)
        # If force is zero, don't move
        valid_moves = force_norms > 1e-8
        forces[valid_moves] /= force_norms[valid_moves, np.newaxis]
        
        # Update positions
        new_c = best_c + step_size * forces
        
        # Clip to cube (strictly inside, with a tiny epsilon to avoid boundary issues in next step)
        new_c = np.clip(new_c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = _compute_radius(new_c)
        
        if new_r > current_r + 1e-7:
            best_c = new_c
            best_r = new_r
            no_improve_count = 0
            # Decay step size slightly
            step_size *= 0.99
        else:
            no_improve_count += 1
            # Reduce step size more aggressively if stuck
            step_size *= 0.95
            if no_improve_count > max_no_improve:
                # Try a small random perturbation to escape local minimum
                if step_size < min_step:
                    break
                perturbation = np.random.normal(0, step_size * 0.5, best_c.shape)
                new_c_pert = best_c + perturbation
                new_c_pert = np.clip(new_c_pert, -0.5 + 1e-9, 0.5 - 1e-9)
                pert_r = _compute_radius(new_c_pert)
                if pert_r > best_r:
                    best_c = new_c_pert
                    best_r = pert_r
                no_improve_count = 0
                step_size = max(step_size, 0.005) # Reset step size to try again

    return best_c, best_r

def _lloyd_init(n_iter=5):
    """
    Initialize using Lloyd's algorithm on a random start to get a better starting configuration.
    Lloyd's algorithm moves each point to the centroid of its Voronoi cell.
    This tends to equalize distances.
    """
    np.random.seed(42)
    c = np.random.uniform(-0.4, 0.4, (N, 3))
    
    for _ in range(n_iter):
        # Compute Voronoi centroids
        # For each point i, find the average position of the region where i is the closest point.
        # Approximation: For each point j, assign it to the closest point i.
        # Then centroid of i is the average of all j assigned to i.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Find closest point for each point j
        closest = np.argmin(dists_sq, axis=1)
        
        # Compute new centroids
        new_c = np.zeros_like(c)
        counts = np.zeros(N)
        
        for i in range(N):
            mask = (closest == i)
            if np.any(mask):
                new_c[i] = np.mean(c[mask], axis=0)
                counts[i] = np.sum(mask)
        
        # If a point has no assigned neighbors (unlikely in 3D with random init), keep it
        zero_counts = (counts == 0)
        new_c[zero_counts] = c[zero_counts]
        
        # Clip to cube
        new_c = np.clip(new_c, -0.45, 0.45)
        
        c = new_c
        
    return c

def construct():
    start_time = time.time()
    
    # 1. Initialization
    # Try a few different initializations:
    # a) Grid
    # b) Lloyd's on random
    # c) Lloyd's on grid
    
    best_c = None
    best_r = 0.0
    
    # Init 1: Grid 5x5x2
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                grid.append([x, y, z])
    c_grid = np.array(grid)
    
    # Init 2: Lloyd's on random
    c_lloyd_random = _lloyd_init(5)
    
    # Init 3: Lloyd's on grid
    # Run Lloyd's on the grid to relax it
    c_lloyd_grid = c_grid.copy()
    for _ in range(5):
        diff = c_lloyd_grid[:, np.newaxis, :] - c_lloyd_grid[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        closest = np.argmin(dists_sq, axis=1)
        new_c = np.zeros_like(c_lloyd_grid)
        counts = np.zeros(N)
        for i in range(N):
            mask = (closest == i)
            if np.any(mask):
                new_c[i] = np.mean(c_lloyd_grid[mask], axis=0)
                counts[i] = np.sum(mask)
        zero_counts = (counts == 0)
        new_c[zero_counts] = c_lloyd_grid[zero_counts]
        c_lloyd_grid = np.clip(new_c, -0.45, 0.45)
        
    inits = [c_grid, c_lloyd_random, c_lloyd_grid]
    
    # 2. Local Refinement
    # We have ~140s. Let's split time among initializations.
    # 3 inits, so ~45s each.
    time_per_init = (DEADLINE - start_time) / 3.0
    
    for i, c_init in enumerate(inits):
        time_left = DEADLINE - start_time
        if time_left < 5:
            break
            
        # Allocate time to this init, leaving 5s for final check
        t_alloc = min(time_per_init, time_left - 5)
        
        c_ref, r_ref = _local_refine(c_init, start_time, start_time + t_alloc)
        
        if r_ref > best_r:
            best_r = r_ref
            best_c = c_ref
            
    # 3. Final Polish with Soft-Min
    # If we have time, do a few steps of a smoother optimization.
    # We can use scipy.optimize.minimize with a soft-min objective if time permits.
    # But our custom local search is usually quite good.
    # Let's check if we can squeeze a bit more out of the best configuration found.
    
    time_left = DEADLINE - start_time
    if time_left > 10 and best_c is not None:
        # Run local refine again on the best result with smaller step size
        # This acts as a polishing step
        c_polish, r_polish = _local_refine(best_c, start_time, DEADLINE - 2)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish
            
    # Ensure valid output
    if best_c is None:
        return c_grid
        
    # Final check
    final_r = _compute_radius(best_c)
    if final_r < 0.1: # Fallback to grid if something went wrong
        return c_grid
        
    return best_c

# To allow testing
if __name__ == "__main__":
    import time
    start = time.time()
    result = construct()
    end = time.time()
    r = _compute_radius(result)
    print(f"Time: {end-start:.2f}s, Radius: {r:.10f}")
