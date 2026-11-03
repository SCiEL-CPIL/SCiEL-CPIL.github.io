"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _compute_score(c):
    """
    Compute the radius: min(min_pairwise_dist/2, min_boundary_dist).
    Vectorized for speed.
    """
    # Boundary distance
    abs_c = np.abs(c)
    boundary_dists = 0.5 - abs_c
    r_bound = np.min(boundary_dists)
    
    # Pairwise distances - compute all pairwise squared distances
    # Use the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    c_sq = np.sum(c**2, axis=1)
    dist_sq = c_sq[:, np.newaxis] + c_sq[np.newaxis, :] - 2.0 * c @ c.T
    # Only need upper triangle
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    # Avoid sqrt if possible - we just need min
    min_d_sq = np.min(d_sq)
    r_pair = np.sqrt(max(min_d_sq, 0.0)) / 2.0
    
    return min(r_pair, r_bound)

def _compute_score_with_indices(c):
    """
    Compute score and return (score, min_pair_idx, min_boundary_idx)
    for local refinement.
    """
    abs_c = np.abs(c)
    boundary_dists = 0.5 - abs_c
    min_bound_idx = np.argmin(boundary_dists)
    r_bound = boundary_dists.flat[min_bound_idx]
    
    c_sq = np.sum(c**2, axis=1)
    dist_sq = c_sq[:, np.newaxis] + c_sq[np.newaxis, :] - 2.0 * c @ c.T
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    min_d_sq_idx = np.argmin(d_sq)
    min_d_sq = d_sq.flat[min_d_sq_idx]
    r_pair = np.sqrt(max(min_d_sq, 0.0)) / 2.0
    
    score = min(r_pair, r_bound)
    
    # Map flat index back to pair indices
    # For upper triangle with N elements, the k-th element (0-indexed) in the flattened upper triangle
    # corresponds to specific i,j. We can precompute or just use the iu arrays.
    i_idx = iu[0][min_d_sq_idx]
    j_idx = iu[1][min_d_sq_idx]
    
    return score, (i_idx, j_idx), min_bound_idx

def _initial_fcc_like():
    """
    Create an initial configuration based on FCC lattice structure,
    which is known to be efficient for sphere packing.
    """
    # FCC lattice in a cube: 4 spheres per unit cell
    # For 100 spheres, use roughly 4^3 = 64 cells, but we need 100
    # Let's use a 5x5x5 grid with FCC offsets, then select 100
    
    # Better approach: use a body-centered or face-centered arrangement
    # Let's create points on an FCC-like grid
    
    # FCC basis vectors in cubic cell of side a:
    # (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    
    # For a cube of side 1, if we use n cells per side, the cell size is 1/n
    # FCC packing fraction is ~0.74
    
    # Let's try n=5, giving 5^3 * 4 = 500 potential sites, pick 100
    n = 5
    cell_size = 1.0 / n
    a = cell_size
    
    points = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                base = np.array([
                    (ix + 0.5) * a - 0.5,
                    (iy + 0.5) * a - 0.5,
                    (iz + 0.5) * a - 0.5
                ])
                # FCC offsets
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5*a, 0.5*a]),
                    np.array([0.5*a, 0, 0.5*a]),
                    np.array([0.5*a, 0.5*a, 0])
                ]
                for off in offsets:
                    p = base + off
                    # Keep only if inside cube
                    if np.all(p >= -0.5 + 1e-9) and np.all(p <= 0.5 - 1e-9):
                        points.append(p)
    
    points = np.array(points)
    
    # If we have more than N, randomly select N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points
        while len(points) < N:
            p = np.random.uniform(-0.4, 0.4, 3)
            points = np.vstack([points, p.reshape(1, -1)])
    
    return points[:N].copy()

def _initial_random():
    """Random initial configuration."""
    return np.random.uniform(-0.4, 0.4, (N, 3))

def _initial_grid():
    """Create an initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _simulated_annealing_with_swaps(c, max_iters=30000, T0=0.01, T1=1e-6, alpha=0.9998, step_scale=0.02, swap_prob=0.3):
    """
    Simulated annealing with both perturbation and swap moves.
    """
    c = c.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    rng = np.random.default_rng()
    
    for it in range(max_iters):
        move_type = rng.random()
        
        if move_type < swap_prob:
            # Swap positions of two random spheres
            i, j = rng.choice(N, 2, replace=False)
            c_new = c.copy()
            c_new[i], c_new[j] = c[j].copy(), c[i].copy()
        elif move_type < swap_prob + 0.4:
            # Move one random point
            idx = rng.integers(N)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + rng.standard_normal(3) * step_scale
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        elif move_type < swap_prob + 0.7:
            # Move two random points
            idxs = rng.choice(N, 2, replace=False)
            c_new = c.copy()
            for idx in idxs:
                c_new[idx] = c_new[idx] + rng.standard_normal(3) * step_scale
                c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        else:
            # Move three random points (larger perturbation)
            idxs = rng.choice(N, 3, replace=False)
            c_new = c.copy()
            for idx in idxs:
                c_new[idx] = c_new[idx] + rng.standard_normal(3) * step_scale * 0.7
                c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _compute_score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            delta_score = new_score - current_score
            prob = math.exp(delta_score / T)
            if rng.random() < prob:
                c = c_new
                current_score = new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Cool down
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _local_refine_fast(c, max_iters=20000, move_size=0.005):
    """
    Fast local refinement using the closest pair and boundary constraints.
    """
    c = c.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    no_improve_count = 0
    
    for it in range(max_iters):
        score, (i, j), min_bound_idx = _compute_score_with_indices(c)
        
        if score > current_score + 1e-15:
            current_score = score
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > 500:
            break
        
        r_pair = _compute_score(c)  # quick check
        abs_c = np.abs(c)
        boundary_dists = 0.5 - abs_c
        r_bound = np.min(boundary_dists)
        
        if r_pair <= r_bound + 1e-12:
            # Pairwise distance is limiting
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try several move sizes
            for move in [move_size, move_size * 0.5, move_size * 0.1]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                new_score = _compute_score(c_new)
                if new_score > current_score + 1e-15:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        else:
            # Boundary is limiting
            min_bound_flat = np.argmin(boundary_dists)
            pt_idx = min_bound_flat // 3
            axis = min_bound_flat % 3
            
            sign = 1 if c[pt_idx, axis] > 0 else -1
            for move in [move_size, move_size * 0.5, move_size * 0.1]:
                c_new = c.copy()
                c_new[pt_idx, axis] = np.clip(c_new[pt_idx, axis] + sign * move, -0.5, 0.5)
                new_score = _compute_score(c_new)
                if new_score > current_score + 1e-15:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    np.random.seed(42)
    random.seed(42)
    
    best_c = None
    best_score = -1.0
    
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    # Try multiple initial configurations
    initializations = [
        _initial_fcc_like,
        _initial_grid,
        _initial_random
    ]
    
    for init_fn in initializations:
        if time.time() - start_time > time_budget:
            break
        
        c = init_fn()
        
        # Phase 1: Coarse simulated annealing
        c, score = _simulated_annealing_with_swaps(
            c, max_iters=20000, T0=0.01, T1=1e-5, alpha=0.9998, step_scale=0.025, swap_prob=0.3
        )
        
        # Phase 2: Local refinement
        c, score = _local_refine_fast(c, max_iters=15000, move_size=0.005)
        
        # Phase 3: Fine simulated annealing
        if time.time() - start_time < time_budget:
            c2, score2 = _simulated_annealing_with_swaps(
                c, max_iters=15000, T0=0.005, T1=1e-6, alpha=0.9999, step_scale=0.01, swap_prob=0.2
            )
            if score2 > score:
                c, score = c2, score2
        
        # Phase 4: Final local refinement
        if time.time() - start_time < time_budget:
            c, score = _local_refine_fast(c, max_iters=10000, move_size=0.002)
        
        if score > best_score:
            best_score = score
            best_c = c.copy()
        
        # If we're getting close to the known best, do more refinement
        if best_score > 0.111:
            if time.time() - start_time < time_budget:
                c, score = _local_refine_fast(c, max_iters=20000, move_size=0.001)
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
    
    # Final check: ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
