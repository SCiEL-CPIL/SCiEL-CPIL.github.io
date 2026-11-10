"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)


def _bottleneck_refine(pts, max_iter=2000, deadline=None):
    """
    Directed refinement: identify the bottleneck (closest pair or wall) and
    apply targeted moves to alleviate it.
    """
    pts = pts.copy()
    n = len(pts)
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    step = 0.05
    decay = 0.97
    no_improve_count = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute distances and identify bottleneck
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pair
        flat_idx = np.argmin(dists)
        i, j = np.unravel_index(flat_idx, (n, n))
        dmin = dists[i, j]
        
        # Find wall bottleneck
        wall_dists = 0.5 - np.abs(pts)
        wmin = np.min(wall_dists)
        wall_idx = np.argmin(wall_dists)
        wall_i, wall_k = np.unravel_index(wall_idx, (n, 3))
        
        current_r = min(dmin / 2.0, wmin)
        
        # Determine which is the bottleneck
        if dmin / 2.0 < wmin:
            # Pair bottleneck: move i and j apart
            direction = pts[i] - pts[j]
            dist = np.linalg.norm(direction)
            if dist < 1e-10:
                direction = np.random.randn(3)
                dist = np.linalg.norm(direction)
            direction /= dist
            
            # Move i away from j, j away from i
            pts[i] += step * direction
            pts[j] -= step * direction
        else:
            # Wall bottleneck: move wall_i away from wall
            axis = wall_k
            if pts[wall_i, axis] > 0:
                pts[wall_i, axis] -= step
            else:
                pts[wall_i, axis] += step
        
        # Clip to cube
        pts = np.clip(pts, -0.4999, 0.4999)
        
        # Check for invalid states
        if not np.all(np.isfinite(pts)):
            pts = best_pts.copy()
            step *= decay
            no_improve_count += 1
            continue
        
        new_score = _compute_score(pts)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            step *= decay
        
        if step < 1e-8 or no_improve_count > 200:
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
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
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
    
    # Optimize each config with bottleneck refinement
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Run bottleneck refinement
        pts_opt, score = _bottleneck_refine(init_pts, max_iter=3000, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
        
        # If we have time, run a second refinement pass on the best result
        if time.time() < deadline - 10:
            pts_opt2, score2 = _bottleneck_refine(pts_opt, max_iter=2000, deadline=deadline)
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
