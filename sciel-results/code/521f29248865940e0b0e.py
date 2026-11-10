"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    # Min distance between any two centers
    # Vectorized distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    # Min distance to walls
    # pts is (N, 3)
    # distance to wall for each coord
    # 0.5 - abs(pts)
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)

def _optimize_repulsion(pts, max_iter=5000, deadline=None):
    """
    Optimize positions by minimizing a repulsive potential energy.
    Energy = sum_{i<j} 1/d^2 + sum_{i,k} 1/(wall_dist)^2
    """
    pts = pts.copy()
    n = len(pts)
    
    # Pre-allocate
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Step size schedule
    step = 0.01
    decay = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute forces
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, 1.0)
        
        # Repulsive force: F_ij = (c_i - c_j) / d^3
        # Sum over j for each i
        # factor = 1 / d^3
        factors = 1.0 / (dists_sq * dists)
        np.fill_diagonal(factors, 0.0)
        
        forces = np.sum(factors[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces
        # Potential U_wall = sum 1/(0.5 - |x|)^2
        # dU/dx = 2 / (0.5 - |x|)^3 * sign(x)
        # If x > 0, dU/dx = 2 / (0.5 - x)^3. Force = -grad U = -2 / (0.5 - x)^3
        # If x < 0, dU/dx = -2 / (0.5 + x)^3. Force = 2 / (0.5 + x)^3
        
        wall_dist = 0.5 - np.abs(pts)
        # Add epsilon to avoid singularity
        wall_dist_safe = np.maximum(wall_dist, 1e-6)
        
        # Gradient of wall potential
        # U = 1/wall_dist^2
        # dU/dx = -2 * wall_dist^-3 * d(wall_dist)/dx
        # d(wall_dist)/dx = -sign(x)
        # dU/dx = 2 * sign(x) / wall_dist^3
        # Force = -dU/dx = -2 * sign(x) / wall_dist^3
        
        signs = np.sign(pts)
        wall_forces = -2.0 * signs / (wall_dist_safe**3)
        
        # Scale wall forces to be comparable to pair forces
        # Pair forces are ~ 1/d^3. If d ~ 0.1, F ~ 1000.
        # Wall forces if wall_dist ~ 0.01, F ~ 2/0.0001 = 20000.
        # They are roughly comparable if distances are similar.
        
        total_forces = forces + wall_forces
        
        # Normalize forces to prevent explosion
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        total_forces = total_forces / norms
        
        # Update positions
        new_pts = pts + step * total_forces
        
        # Clamp to cube (slightly inside to avoid infinite forces)
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        # Check score
        new_score = _compute_score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If no improvement, reduce step size
            step *= decay
        
        # Early stopping if step is too small
        if step < 1e-8:
            break
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initial configurations
    configs = []
    
    # 1. Jittered 4x4x4 grid (64 points, take 50)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to get a random subset? Or just take first 50.
    # Taking first 50 from a 4x4x4 grid leaves holes.
    # Better to jitter the whole grid and take 50? No, we need 50 points.
    # Let's just take the first 50 and jitter them.
    p1 = grid[:N].copy()
    p1 += np.random.randn(N, 3) * 0.05
    configs.append(p1)
    
    # 2. Jittered 3x4x5 grid (60 points, take 50)
    # Create 3x4x5 grid
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
    # This is complex to generate exactly 50. Skip for now, grids are better.
    
    # Optimize each config
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Allocate remaining time / remaining configs
        remaining_configs = len(configs) - idx
        time_per_config = (deadline - time.time()) / remaining_configs
        
        # Run optimization
        # We can run multiple iterations of the optimizer or just one long run.
        # Let's run one long run with a deadline.
        pts_opt, score = _optimize_repulsion(init_pts, max_iter=10000, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt
    
    # Fallback if all failed or score is bad
    if best_pts is None or best_score < 0.1:
        # Simple grid fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _compute_score(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
