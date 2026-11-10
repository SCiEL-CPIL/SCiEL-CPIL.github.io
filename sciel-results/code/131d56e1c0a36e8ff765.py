"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    # Min distance between any two centers
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    # Min distance to walls
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)


def _optimize_repulsion(pts, max_iter=5000, deadline=None):
    """
    Optimize positions by minimizing a repulsive potential energy.
    Energy = sum_{i<j} 1/d^2 + sum_{i,k} 1/(wall_dist)^2
    """
    pts = pts.copy()
    n = len(pts)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    step = 0.01
    decay = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, 1.0)
        
        factors = 1.0 / (dists_sq * dists)
        np.fill_diagonal(factors, 0.0)
        
        forces = np.sum(factors[:, :, np.newaxis] * diff, axis=1)
        
        wall_dist = 0.5 - np.abs(pts)
        wall_dist_safe = np.maximum(wall_dist, 1e-6)
        
        signs = np.sign(pts)
        wall_forces = -2.0 * signs / (wall_dist_safe**3)
        
        total_forces = forces + wall_forces
        
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        total_forces = total_forces / norms
        
        new_pts = pts + step * total_forces
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_score = _compute_score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            step *= decay
        
        if step < 1e-8:
            break
            
    return best_pts, best_score


def _optimize_local(pts, max_iter=2000, deadline=None):
    """
    Local optimization: push spheres apart using a steepest-descent-like approach
    on the minimum distance. This refines the packing by directly targeting the
    bottleneck (closest pair or wall).
    """
    pts = pts.copy()
    n = len(pts)
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    step = 0.05
    decay = 0.95
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Identify the bottleneck: is it a pair or a wall?
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        dmin = np.min(dists)
        
        wall_dists = 0.5 - np.abs(pts)
        wmin = np.min(wall_dists)
        
        current_r = min(dmin / 2.0, wmin)
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pair forces: push apart all pairs, weighted by proximity
        # Use a soft potential to avoid singularities
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(3) * 0.01
                    dist = np.linalg.norm(d)
                # Force magnitude: stronger when closer
                # Use 1/dist^2 for strong repulsion at close range
                force_mag = 1.0 / (dist**2 + 1e-6)
                force_vec = d / dist * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Wall forces: push away from walls
        for i in range(n):
            for k in range(3):
                margin = 0.5 - abs(pts[i, k])
                if margin < 0.1:  # Only act when close to wall
                    if pts[i, k] > 0:
                        forces[i, k] -= 1.0 / (margin**2 + 1e-6)
                    else:
                        forces[i, k] += 1.0 / (margin**2 + 1e-6)
        
        # Normalize forces
        fnorm = np.linalg.norm(forces, axis=1, keepdims=True)
        fnorm = np.maximum(fnorm, 1e-10)
        forces = forces / fnorm
        
        # Update positions
        new_pts = pts + step * forces
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_score = _compute_score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            step *= decay
            if step < 1e-6:
                break
    
    return best_pts, best_score


def _generate_initial_configs():
    """Generate diverse initial configurations for optimization."""
    configs = []
    np.random.seed(42)
    
    # 1. Jittered 4x4x4 grid (64 points, take 50)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    p1 = grid[:N].copy()
    p1 += np.random.randn(N, 3) * 0.05
    configs.append(p1)
    
    # 2. Jittered 3x4x5 grid (60 points, take 50)
    m1, m2, m3 = 3, 4, 5
    t1 = (np.arange(m1) + 0.5) / m1 - 0.5
    t2 = (np.arange(m2) + 0.5) / m2 - 0.5
    t3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid2 = np.array([(x, y, z) for z in t3 for y in t2 for x in t1])
    p2 = grid2[:N].copy()
    p2 += np.random.randn(N, 3) * 0.05
    configs.append(p2)
    
    # 3. Random points
    p3 = np.random.uniform(-0.4, 0.4, (N, 3))
    configs.append(p3)
    
    # 4. FCC-like structure (subset)
    # FCC lattice points in unit cell: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # Scale to fit in [-0.5, 0.5]
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                fcc_points.append((i - 1, j - 1, k - 1))
                # Face centers
                fcc_points.append((i - 1, j - 0.5, k - 0.5))
                fcc_points.append((i - 0.5, j - 1, k - 0.5))
                fcc_points.append((i - 0.5, j - 0.5, k - 1))
    fcc_points = np.array(fcc_points)
    # Scale to fit in [-0.5, 0.5]
    fcc_points *= 1.0 / 3.0
    # Shuffle and take 50
    np.random.shuffle(fcc_points)
    p4 = fcc_points[:N].copy()
    p4 += np.random.randn(N, 3) * 0.02
    configs.append(p4)
    
    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_pts = None
    best_score = -1.0
    
    # Generate initial configurations
    configs = _generate_initial_configs()
    
    # Optimize each config with both methods
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Method 1: Repulsion-based optimization
        pts_opt1, score1 = _optimize_repulsion(init_pts, max_iter=3000, deadline=deadline)
        
        if score1 > best_score:
            best_score = score1
            best_pts = pts_opt1.copy()
        
        # Method 2: Local optimization on the result of Method 1
        if time.time() < deadline:
            pts_opt2, score2 = _optimize_local(pts_opt1, max_iter=1000, deadline=deadline)
            
            if score2 > best_score:
                best_score = score2
                best_pts = pts_opt2.copy()
    
    # Fallback if all failed or score is bad
    if best_pts is None or best_score < 0.1:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _compute_score(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
