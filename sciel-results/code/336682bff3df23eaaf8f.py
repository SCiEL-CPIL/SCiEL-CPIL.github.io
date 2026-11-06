import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0 or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Distance to walls
    # min(0.5 - x, x + 0.5) for each coordinate, then min over all points and coords
    wall_dist = np.min(np.minimum(0.5 - centers, centers + 0.5))
    
    if wall_dist <= 0:
        return 0.0

    # Distance to neighbors
    # Vectorized pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N):
    """Generate a hexagonal grid of points."""
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _force_relaxation(centers, time_limit, start_time, max_iters=5000, seed=42):
    """
    Simultaneous force-directed relaxation.
    Instead of fixing the single worst constraint, apply repulsive forces
    from ALL pairs that are closer than a threshold (e.g., 1.5 * current_r)
    and ALL walls that are close.
    This allows the system to "breathe" and find better local minima.
    """
    centers = centers.copy()
    n = len(centers)
    
    np.random.seed(seed)
    
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    # Parameters
    force_cutoff_mult = 1.5  # Apply force if dist < cutoff_mult * current_r
    lr = 0.05
    decay = 0.999
    stall_count = 0
    last_best = best_score
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # 1. Compute distances
        dist_mat = cdist(centers, centers)
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        
        # Wall distances
        wall_dists = np.minimum(0.5 - centers, centers + 0.5)
        min_wall_dist = np.min(wall_dists)
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_score + 1e-9:
            best_score = current_r
            best_centers = centers.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > 50:
                # Restart with random perturbation if stuck
                centers = best_centers + np.random.normal(0, 0.01, (n, 2))
                centers = np.clip(centers, -0.49, 0.49)
                stall_count = 0
                continue

        # 2. Calculate Forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Only consider pairs within cutoff
        cutoff = force_cutoff_mult * current_r
        # Mask for pairs closer than cutoff
        mask = dist_mat < cutoff
        # Exclude diagonal (already inf)
        
        # Vectorized force calculation:
        # F_i = sum_j (vec_ij / |vec_ij|^3) * strength
        # We want to push them apart.
        
        # To avoid O(N^2) memory explosion for large N in Python loops,
        # we can do it in chunks or rely on numpy broadcasting if N is small enough.
        # N=200 -> 40k pairs. Broadcasting creates a 200x200x2 array. That's fine.
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e-9) # Avoid div by zero
        
        # Inverse cube law for force magnitude (soft repulsion)
        # F ~ 1/d^2 direction. Let's use F ~ 1/d^3 * vec so magnitude is 1/d^2.
        # Or simpler: F ~ (cutoff - d) * unit_vec? Linear spring?
        # Let's use a smooth repulsion: F = (cutoff^2 / d^2 - 1) * unit_vec if d < cutoff
        # Actually, standard Lennard-Jones or just 1/d^2 is robust.
        
        # Let's use a simple repulsive force proportional to 1/dist^2
        # But only for pairs closer than cutoff to save computation? 
        # Numpy broadcasting is fast enough for 200.
        
        # Mask out pairs that are far away to reduce noise?
        # If we apply force to all, it's a global expansion. 
        # It's better to only apply to local neighbors to maintain structure.
        
        force_mag = np.zeros_like(dist_sq)
        active = (dist_mat < cutoff) & (dist_mat > 1e-6)
        # Force magnitude = 1 / dist^2
        force_mag[active] = 1.0 / (dist_sq[active] + 1e-9)
        
        # Force vector = force_mag * diff
        # Sum over neighbors
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces
        # Push away from walls if close
        wall_cutoff = force_cutoff_mult * current_r
        
        # Left wall: x < -0.5 + wall_cutoff
        # Right wall: x > 0.5 - wall_cutoff
        # Bottom: y < -0.5 + wall_cutoff
        # Top: y > 0.5 - wall_cutoff
        
        # Force proportional to (wall_cutoff - dist_to_wall)^2 or similar
        # Let's use linear push: F = (wall_cutoff - dist_to_wall)
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Left
        dist_left = x + 0.5
        active_left = dist_left < wall_cutoff
        forces[active_left, 0] += (wall_cutoff - dist_left[active_left])
        
        # Right
        dist_right = 0.5 - x
        active_right = dist_right < wall_cutoff
        forces[active_right, 0] -= (wall_cutoff - dist_right[active_right])
        
        # Bottom
        dist_bottom = y + 0.5
        active_bottom = dist_bottom < wall_cutoff
        forces[active_bottom, 1] += (wall_cutoff - dist_bottom[active_bottom])
        
        # Top
        dist_top = 0.5 - y
        active_top = dist_top < wall_cutoff
        forces[active_top, 1] -= (wall_cutoff - dist_top[active_top])
        
        # 3. Update positions
        # Normalize forces?
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-9:
            # Scale step
            step = lr / max_force
            centers += forces * step
            
        # Project to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay learning rate slightly
        lr *= decay

    return best_centers, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    # 1. Generate diverse initializations
    inits = []
    
    # Hex Grid
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    # Square Grid
    square_grid = _generate_square_grid(N)
    inits.append(square_grid)
    
    # Jittered Hex
    if hex_grid is not None:
        np.random.seed(42)
        jittered_hex = hex_grid + np.random.normal(0, 0.01, hex_grid.shape)
        jittered_hex = np.clip(jittered_hex, -0.49, 0.49)
        inits.append(jittered_hex)
        
    # Random Initializations
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)
        
    # Sort initializations by their initial score (best first)
    scored_inits = []
    for init in inits:
        score = _compute_score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Run force relaxation on each initialization
    for idx, (score, init_centers) in enumerate(scored_inits):
        if time.time() - start_time > time_limit:
            break
            
        # Use a different seed for each run to vary restarts
        seed = 100 + idx
        centers, local_score = _force_relaxation(init_centers, time_limit, start_time, seed=seed)
        
        if local_score > best_score:
            best_score = local_score
            best_centers = centers
            
    # 3. If time remains, do random restarts from the best found so far
    if best_centers is not None and time.time() - start_time < time_limit - 10:
        restart_count = 0
        while time.time() - start_time < time_limit - 2:
            restart_count += 1
            seed = 1000 + restart_count
            np.random.seed(seed)
            # Perturb best solution
            perturbation_scale = 0.01 + 0.01 * (restart_count % 3) # Vary scale
            perturbed = best_centers + np.random.normal(0, perturbation_scale, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            centers, local_score = _force_relaxation(perturbed, time_limit, start_time, seed=seed)
            
            if local_score > best_score:
                best_score = local_score
                best_centers = centers
                
    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
