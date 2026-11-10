"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    if n < 2:
        return 0.5
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)

def _force_calc(pts, r_target=0.135):
    """
    Calculate repulsive forces between all pairs and wall forces.
    Returns force array of shape (N, 3).
    Uses a soft-sphere potential: F ~ 1/d^2 for d < 2*r_target, else 0.
    Wall forces push spheres inward if they are too close to walls.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion: F = 1/d^2 along direction (scaled)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only interact if distance < 2*r_target (overlap or close)
    mask = dists < 2.5 * r_target
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    
    # Force magnitude: strong repulsion when close
    # F = k / d^2, direction = diff / d
    # We want to maximize min distance, so we push apart when d < 2r
    # Use a smooth potential: V = 1/d^2, F = -grad V = 2/d^3 * r_hat
    # Simplified: F = 1/d^2 along r_hat
    force_mag = np.zeros_like(dists)
    force_mag[mask] = 1.0 / (dists_safe[mask] ** 2)
    
    # Force vector = force_mag * (diff / dist)
    # diff / dist is the unit vector from j to i
    unit_vectors = diff / dists_safe[:, :, np.newaxis]
    
    # Sum forces from all neighbors
    forces = np.sum(force_mag[:, :, np.newaxis] * unit_vectors, axis=1)
    
    # Wall forces: push inward if close to wall
    # Distance to wall: 0.5 - |x|
    # If 0.5 - |x| < r_target, push inward
    for axis in range(3):
        dist_to_wall = 0.5 - np.abs(pts[:, axis])
        # If close to wall, push toward center
        # Sign of push: if x > 0, push left (-x); if x < 0, push right (+x)
        # So force = -sign(x) * strength
        close_to_wall = dist_to_wall < r_target
        if np.any(close_to_wall):
            strength = 5.0 / (dist_to_wall[close_to_wall] ** 2)
            signs = np.sign(pts[close_to_wall, axis])
            # Push away from wall: if x>0, force is negative; if x<0, force is positive
            forces[close_to_wall, axis] -= signs * strength
    
    return forces

def _anneal(pts, deadline, r_target=0.135):
    """
    Simulated annealing with force-directed dynamics.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Initial temperature
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    step_size = 0.02
    no_improve = 0
    
    it = 0
    while time.time() < deadline:
        it += 1
        
        # Calculate forces
        F = _force_calc(pts, r_target)
        
        # Random noise (Langevin dynamics)
        noise = np.random.randn(n, 3) * step_size * np.sqrt(T)
        
        # Update positions
        pts_new = pts + F * step_size + noise
        
        # Clip to cube (slightly inside to avoid boundary issues)
        pts_new = np.clip(pts_new, -0.499, 0.499)
        
        # Compute scores
        score_old = _compute_score(pts)
        score_new = _compute_score(pts_new)
        
        # Metropolis acceptance
        if score_new >= score_old:
            pts = pts_new
        else:
            # Accept with probability exp((score_new - score_old) / T)
            delta = score_new - score_old
            prob = np.exp(delta / (T * 0.1))  # Scale delta
            if np.random.rand() < prob:
                pts = pts_new
        
        # Update best
        if score_new > best_score + 1e-10:
            best_score = score_new
            best_pts = pts_new.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If stuck, restart with higher T
            if no_improve > 500:
                T = 0.02
                no_improve = 0
        
        # If no improvement for a long time, perturb
        if no_improve > 1000:
            pts = best_pts.copy()
            pts += np.random.randn(n, 3) * 0.05
            pts = np.clip(pts, -0.499, 0.499)
            no_improve = 0
            T = 0.03
    
    return best_pts, best_score

def _polish(pts, deadline):
    """
    High-precision local refinement using gradient descent on the minimum distance.
    """
    pts = pts.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    step = 0.01
    decay = 0.95
    no_improve = 0
    
    while time.time() < deadline:
        # Compute all distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance pair
        flat_idx = np.argmin(dists)
        i, j = np.unravel_index(flat_idx, (n, n))
        dmin = dists[i, j]
        
        # Find wall constraint
        wall_dists = 0.5 - np.abs(pts)
        wmin = np.min(wall_dists)
        wall_flat_idx = np.argmin(wall_dists)
        wall_i, wall_k = np.unravel_index(wall_flat_idx, (n, 3))
        
        current_r = min(dmin / 2.0, wmin)
        
        # Determine bottleneck
        if dmin / 2.0 < wmin:
            # Pair bottleneck
            direction = pts[i] - pts[j]
            dist = np.linalg.norm(direction)
            if dist < 1e-10:
                direction = np.random.randn(3)
                dist = np.linalg.norm(direction)
            direction /= dist
            
            pts[i] += step * direction
            pts[j] -= step * direction
        else:
            # Wall bottleneck
            axis = wall_k
            if pts[wall_i, axis] > 0:
                pts[wall_i, axis] -= step
            else:
                pts[wall_i, axis] += step
        
        pts = np.clip(pts, -0.4999, 0.4999)
        
        new_score = _compute_score(pts)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            step *= decay
        
        if step < 1e-8 or no_improve > 100:
            break
    
    return best_pts, best_score

def _generate_initial_configs():
    """Generate diverse structured initial configurations."""
    configs = []
    np.random.seed(42)
    
    # 1. Simple cubic grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    p1 = grid[:N].copy()
    p1 += np.random.randn(N, 3) * 0.02
    configs.append(p1)
    
    # 2. FCC-like structure
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fcc_points.append((i - 1, j - 1, k - 1))
                fcc_points.append((i - 1, j - 0.5, k - 0.5))
                fcc_points.append((i - 0.5, j - 1, k - 0.5))
                fcc_points.append((i - 0.5, j - 0.5, k - 1))
    fcc_points = np.array(fcc_points)
    fcc_points *= 1.0 / 3.0
    np.random.shuffle(fcc_points)
    p2 = fcc_points[:N].copy()
    p2 += np.random.randn(N, 3) * 0.01
    configs.append(p2)
    
    # 3. BCC-like structure
    bcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bcc_points.append((i - 1, j - 1, k - 1))
                bcc_points.append((i - 0.5, j - 0.5, k - 0.5))
    bcc_points = np.array(bcc_points)
    bcc_points *= 1.0 / 3.0
    np.random.shuffle(bcc_points)
    p3 = bcc_points[:N].copy()
    p3 += np.random.randn(N, 3) * 0.01
    configs.append(p3)
    
    # 4. Random with minimum separation
    p4 = np.random.uniform(-0.45, 0.45, (N, 3))
    configs.append(p4)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_pts = None
    best_score = -1.0
    
    # Generate initial configurations
    configs = _generate_initial_configs()
    
    # Optimize each config with annealing + polish
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Run annealing
        pts_annealed, score_annealed = _anneal(init_pts, deadline - 5)
        
        if score_annealed > best_score:
            best_score = score_annealed
            best_pts = pts_annealed.copy()
        
        # If we have time, run polish on the best result
        if time.time() < deadline - 10:
            pts_polished, score_polished = _polish(pts_annealed, deadline - 2)
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished.copy()
    
    # Fallback if all failed
    if best_pts is None or best_score < 0.1:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _compute_score(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
