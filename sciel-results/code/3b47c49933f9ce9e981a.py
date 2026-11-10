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
    # Min distance between any two centers
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    # Min distance to walls
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)

def _force_calc(pts):
    """
    Calculate repulsive forces between all pairs and wall forces.
    Returns force array of shape (N, 3).
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion: F = 1/d^2 along direction
    # We use a softening parameter to avoid division by zero and numerical instability
    # but since we are maximizing min distance, strong repulsion is good.
    # Using 1/d^3 * dir vector is gradient of 1/d.
    # Let's use gradient of 1/d^2 which is -2/d^3 * dir.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid div by 0 on diag
    
    # Force magnitude factor: d / d^6 = 1/d^5 ? No.
    # Potential U = 1/d^2. F = -grad U = 2 * d_vec / d^5.
    # Let's just use a strong repulsion: F_vec = diff / dist^3 (gradient of 1/dist)
    # Or even stronger for packing: F_vec = diff / dist^5 (gradient of 1/dist^2)
    
    # Safe calculation
    dists = np.sqrt(dists_sq)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Calculate forces
    # F_ij on i due to j: (p_i - p_j) / |p_i - p_j|^3 * strength
    # We want to push them apart.
    
    inv_d3 = np.zeros_like(dists)
    inv_d3[mask] = 1.0 / (dists[mask] ** 3)
    
    # Force on i from j is diff[i,j] * inv_d3[i,j]
    # Sum over j
    forces += np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
    
    # Wall forces
    # If close to wall, push inward
    # Wall distance: 0.5 - |x|
    # Force is proportional to 1/wall_dist^2 if close
    margin = 0.1
    for k in range(3):
        # Positive x wall
        d_pos = 0.5 - pts[:, k]
        # Negative x wall
        d_neg = 0.5 + pts[:, k]
        
        # Force from +x wall (pushes in -x direction if close)
        # F = -sign(x) * 1/d^2
        # Let's use a smooth potential
        # If d < margin, apply force
        f_pos = np.zeros(n)
        close_pos = d_pos < margin
        if np.any(close_pos):
            f_pos[close_pos] = -1.0 / (d_pos[close_pos] ** 2)
            
        f_neg = np.zeros(n)
        close_neg = d_neg < margin
        if np.any(close_neg):
            f_neg[close_neg] = 1.0 / (d_neg[close_neg] ** 2)
            
        forces[:, k] += f_pos + f_neg
        
    return forces

def _anneal(pts_init, max_iter=2000, deadline=None):
    """
    Simulated Annealing to maximize minimum distance.
    """
    pts = pts_init.copy()
    n = len(pts)
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    T = 0.05
    T_min = 0.0001
    alpha = 0.995 # Cooling rate
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Calculate forces
        F = _force_calc(pts)
        
        # Add noise
        noise = np.random.randn(n, 3) * T
        
        # Update
        pts_new = pts + 0.1 * (F + noise)
        
        # Keep inside cube (hard constraint)
        pts_new = np.clip(pts_new, -0.499, 0.499)
        
        # Evaluate
        score_new = _compute_score(pts_new)
        score_old = _compute_score(pts)
        
        # Accept if better or with probability exp((new-old)/T)
        # Here 'score' is higher is better, so delta = score_new - score_old
        if score_new > score_old:
            pts = pts_new
            if score_new > best_score:
                best_score = score_new
                best_pts = pts_new.copy()
        else:
            delta = score_new - score_old
            if delta > 0:
                pass
            else:
                # Metropolis criterion
                if np.random.random() < np.exp(delta / T):
                    pts = pts_new
                    
        T *= alpha
        if T < T_min:
            break
            
    return best_pts, best_score

def _polish(pts, max_iter=500, deadline=None):
    """
    Local search specifically targeting the bottleneck constraints.
    """
    pts = pts.copy()
    n = len(pts)
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    step = 0.01
    no_improve = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Identify bottleneck
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        dmin = np.min(dists)
        
        wall_dists = 0.5 - np.abs(pts)
        wmin = np.min(wall_dists)
        
        current_r = min(dmin / 2.0, wmin)
        
        if dmin / 2.0 < wmin:
            # Pair bottleneck
            flat_idx = np.argmin(dists)
            i, j = np.unravel_index(flat_idx, (n, n))
            direction = pts[i] - pts[j]
            dist = np.linalg.norm(direction)
            if dist < 1e-10:
                direction = np.random.randn(3)
                dist = np.linalg.norm(direction)
            direction /= dist
            
            # Try moving i and j apart
            # Randomize direction slightly to escape symmetry
            direction += np.random.randn(3) * 0.1
            direction /= np.linalg.norm(direction)
            
            pts[i] += step * direction
            pts[j] -= step * direction
        else:
            # Wall bottleneck
            wall_idx = np.argmin(wall_dists)
            i, k = np.unravel_index(wall_idx, (n, 3))
            # Move away from wall
            if pts[i, k] > 0:
                pts[i, k] -= step
            else:
                pts[i, k] += step
            # Add small random component in other axes
            pts[i] += np.random.randn(3) * step * 0.1
            pts[i, k] -= np.sign(pts[i, k]) * step * 0.9 # Ensure move towards center
            
        pts = np.clip(pts, -0.499, 0.499)
        
        new_score = _compute_score(pts)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_pts = pts.copy()
            no_improve = 0
            # Increase step slightly if improving? Or keep constant?
            # Let's keep constant for stability in polish phase
        else:
            no_improve += 1
            if no_improve > 50:
                step *= 0.95
                if step < 1e-6:
                    break
                    
    return best_pts, best_score

def _generate_initial_configs():
    """Generate diverse structured initial configurations."""
    configs = []
    np.random.seed(42)
    
    # 1. Simple cubic grid (4x4x4 = 64, take 50)
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
    
    # 4. Random with minimum separation (Poisson disk sampling approximation)
    p4 = np.random.uniform(-0.45, 0.45, (N, 3))
    configs.append(p4)
    
    # 5. Another random start
    p5 = np.random.uniform(-0.45, 0.45, (N, 3))
    configs.append(p5)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_pts = None
    best_score = -1.0
    
    # Generate initial configurations
    configs = _generate_initial_configs()
    
    # Optimize each config
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Phase 1: Annealing to find a good basin
        pts_annealed, score_annealed = _anneal(init_pts, max_iter=3000, deadline=deadline)
        
        # Phase 2: Polishing to refine the bottleneck
        pts_opt, score_opt = _polish(pts_annealed, max_iter=1000, deadline=deadline)
        
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt.copy()
            
        # If we have time, try a second polish on the best result found so far
        if time.time() < deadline - 5 and best_pts is not None:
            pts_opt2, score2 = _polish(best_pts, max_iter=500, deadline=deadline)
            if score2 > best_score:
                best_score = score2
                best_pts = pts_opt2.copy()

    # Fallback if all failed
    if best_pts is None or best_score < 0.1:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _compute_score(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
