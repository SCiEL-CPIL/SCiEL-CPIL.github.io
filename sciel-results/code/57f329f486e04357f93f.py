"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _compute_radius_fast(c, moved_idx):
    """Fast radius computation when only some points moved."""
    n = len(c)
    # Wall distances for all points
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # Pairwise: only need distances involving moved points
    # Plus the minimum among non-moved pairs (precomputed or recomputed)
    # For simplicity with small N, compute all but use cdist
    if len(moved_idx) == 0:
        return min_wall
    
    # Compute distances from moved points to all others
    moved = c[moved_idx]
    others = c
    d = cdist(moved, others)
    # Exclude self-distances
    for i, mi in enumerate(moved_idx):
        d[i, mi] = np.inf
    min_pair_moved = np.min(d)
    
    # Also need min among non-moved pairs
    if len(moved_idx) < n:
        non_moved_mask = np.ones(n, dtype=bool)
        non_moved_mask[moved_idx] = False
        non_moved = c[non_moved_mask]
        if len(non_moved) >= 2:
            d_nm = cdist(non_moved, non_moved)
            np.fill_diagonal(d_nm, np.inf)
            min_pair_nonmoved = np.min(d_nm)
            min_pair = min(min_pair_moved, min_pair_nonmoved)
        else:
            min_pair = min_pair_moved
    else:
        min_pair = min_pair_moved
    
    return min(min_pair / 2.0, min_wall)


def _initial_config(n):
    """Generate initial configuration using a structured grid + perturbation."""
    # Use a 4x4x4 grid = 64 points, select 50
    # Or use FCC-like structure
    # Try a simple approach: place points on a body-centered cubic-like grid
    
    # Generate candidate points on a grid
    pts = []
    # 4x4x4 grid
    for i in range(4):
        for j in range(4):
            for k in range(4):
                p = np.array([i/3.0 - 0.5, j/3.0 - 0.5, k/3.0 - 0.5])
                pts.append(p)
    pts = np.array(pts)
    
    # Add some FCC-like points
    a = 0.25
    offsets = np.array([
        [0, 0, 0], [0, 0.5*a, 0.5*a], [0.5*a, 0, 0.5*a], [0.5*a, 0.5*a, 0]
    ])
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                for off in offsets:
                    p = np.array([i, j, k]) * a + off
                    if np.all(np.abs(p) <= 0.49):
                        pts = np.vstack([pts, [p]])
    
    # Remove duplicates (points too close)
    pts = np.unique(np.round(pts, 6), axis=0)
    
    # Sort by distance from center, take closest n
    d_center = np.sum(pts * pts, axis=1)
    order = np.argsort(d_center)
    pts = pts[order]
    
    if len(pts) < n:
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(pts), 3))
        pts = np.vstack([pts, extra])
    
    return pts[:n].copy()


def _local_refine(c, max_iters=5000, time_limit=None):
    """Local refinement: iteratively move each point to maximize its minimum distance to neighbors and walls."""
    n = len(c)
    c = c.copy()
    improved = True
    iters = 0
    t_start = time.time()
    
    while improved and iters < max_iters:
        if time_limit is not None and time.time() - t_start > time_limit:
            break
        improved = False
        iters += 1
        
        for i in range(n):
            # Find the point's current minimum distance to others and walls
            # Try to move point i to increase its minimum distance
            # Use a simple gradient-like step: move away from nearest neighbor
            diff = c - c[i]
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            dists[i] = np.inf
            nearest_j = np.argmin(dists)
            
            # Wall distances
            wall_dists = 0.5 - np.abs(c[i])
            min_wall = np.min(wall_dists)
            
            # Determine direction to move
            min_dist = min(dists[nearest_j], 2.0 * min_wall)
            
            if dists[nearest_j] < 2.0 * min_wall:
                # Move away from nearest neighbor
                direction = (c[i] - c[nearest_j])
                direction = direction / (np.linalg.norm(direction) + 1e-10)
            else:
                # Move away from wall
                # Find which wall is closest
                min_wall_idx = np.argmin(wall_dists)
                # Determine axis and direction
                # wall_dists[k] = 0.5 - |c[i,k]|
                # If c[i,k] > 0, wall is at +0.5, move in -x direction
                # If c[i,k] < 0, wall is at -0.5, move in +x direction
                axis = min_wall_idx
                if c[i, axis] > 0:
                    direction = np.zeros(3)
                    direction[axis] = -1.0
                else:
                    direction = np.zeros(3)
                    direction[axis] = 1.0
            
            # Try several step sizes
            best_pos = c[i].copy()
            best_min_dist = min_dist
            
            for step in [0.001, 0.005, 0.01, 0.02, 0.05]:
                new_pos = c[i] + direction * step
                new_pos = np.clip(new_pos, -0.499, 0.499)
                
                # Compute new minimum distance for this point
                new_diff = new_pos - c
                new_dists = np.sqrt(np.sum(new_diff * new_diff, axis=1))
                new_dists[i] = np.inf
                new_min_pair = np.min(new_dists)
                new_wall = np.min(0.5 - np.abs(new_pos))
                new_min_dist = min(new_min_pair, 2.0 * new_wall)
                
                if new_min_dist > best_min_dist:
                    best_min_dist = new_min_dist
                    best_pos = new_pos
            
            if not np.allclose(best_pos, c[i]):
                c[i] = best_pos
                improved = True
    
    return c


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s
    
    np.random.seed(42)
    
    # Start with initial configuration
    c = _initial_config(N)
    c += np.random.uniform(-0.01, 0.01, size=c.shape)
    c = np.clip(c, -0.499, 0.499)
    
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # Simulated annealing with fast evaluation
    T = 0.03
    T_min = 1e-7
    cooling = 0.9995
    
    n = N
    step_count = 0
    
    while T > T_min:
        if time.time() > deadline - 20.0:
            break
        
        # Move one random point
        idx = np.random.randint(n)
        old = c[idx].copy()
        
        # Random displacement - scale with current temperature
        scale = T * 1.5
        c[idx] += np.random.normal(0, scale, size=3)
        c[idx] = np.clip(c[idx], -0.499, 0.499)
        
        r_new = _compute_radius_fast(c, [idx])
        
        if r_new >= best_r:
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            # Compute old radius for comparison
            c[idx] = old
            r_old = _compute_radius_fast(c, [idx])
            c[idx] = old + np.random.normal(0, scale, size=3)
            c[idx] = np.clip(c[idx], -0.499, 0.499)
            r_new2 = _compute_radius_fast(c, [idx])
            
            delta = r_new2 - r_old
            if delta / T > np.log(np.random.random() + 1e-15):
                if r_new2 > best_r:
                    best_r = r_new2
                    best_c = c.copy()
            else:
                c[idx] = old
        
        T *= cooling
        step_count += 1
    
    # Local refinement on best configuration
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 2.0:
        refined = _local_refine(best_c, max_iters=2000, time_limit=remaining_time)
        r_refined = _compute_radius(refined)
        if r_refined > best_r:
            best_r = r_refined
            best_c = refined
    
    return best_c
# EVOLVE-BLOCK-END
