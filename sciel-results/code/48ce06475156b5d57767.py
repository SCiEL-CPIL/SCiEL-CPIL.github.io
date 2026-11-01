"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_score(centers):
    """Compute the packing radius efficiently."""
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    tree = cKDTree(centers)
    pairs, dists = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_pair / 2.0, min_wall)


def _sa_run(centers, rng, time_budget):
    """
    Efficient simulated annealing with incremental distance tracking.
    """
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    # Precompute wall distances
    wall = 0.5 - np.abs(centers)
    
    # Precompute distance matrix (squared)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Track min pair distance and which pair
    min_pair_sq = np.min(dist_sq)
    min_wall = np.min(wall)
    cur_score = min(np.sqrt(min_pair_sq) / 2.0, min_wall)
    
    best_centers = centers.copy()
    best_score = cur_score
    
    # SA parameters - tuned for better exploration
    T_start = 0.02
    T_end = 1e-10
    n_iters = 500000
    alpha = (T_end / T_start) ** (1.0 / n_iters)
    
    # Adaptive step size
    step = 0.015
    base_step = 0.015
    no_improve = 0
    max_no_improve = 3000
    
    # Track the two smallest pair distances to enable fast incremental updates
    # Sort all pair distances to find the two smallest
    flat_idx = np.argsort(dist_sq)
    second_min_pair_sq = dist_sq.flat[flat_idx[1]] if n > 2 else np.inf
    
    for it in range(n_iters):
        if it % 1000 == 0 and time.time() - start_time > time_budget:
            break
        
        T = T_start * (alpha ** it)
        
        # Pick random circle
        i = rng.integers(n)
        old_pos = centers[i].copy()
        
        # Generate new position
        new_pos = old_pos + rng.normal(0, step, 2)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Compute new distances from i to all others
        new_dists = np.sum((centers - new_pos) ** 2, axis=1)
        new_dists[i] = np.inf
        
        # New wall distance for circle i
        new_wall_i = np.min(0.5 - np.abs(new_pos))
        old_wall_i = np.min(0.5 - np.abs(old_pos))
        
        # Determine if this move could improve the score
        # New min wall = min(new_wall_i, min of other walls)
        other_walls = np.delete(wall, i)
        min_other_wall = np.min(other_walls) if len(other_walls) > 0 else np.inf
        new_min_wall = min(new_wall_i, min_other_wall)
        
        # New min pair involving i
        new_min_pair_inv_i = np.min(new_dists)
        
        # The true new min pair is min(new_min_pair_inv_i, min pair not involving i)
        # We can approximate: if old min pair didn't involve i, it's still valid
        # If it did involve i, we need to find the next smallest
        
        # Quick check: upper bound on new score
        upper_bound = min(np.sqrt(new_min_pair_inv_i) / 2.0, new_min_wall)
        
        if upper_bound > cur_score:
            # Potentially improving move - apply and verify
            centers[i] = new_pos
            wall[i] = new_wall_i
            
            # Update distance matrix row/col for i
            diff_i = centers - new_pos
            dist_sq[i, :] = np.sum(diff_i ** 2, axis=1)
            dist_sq[:, i] = dist_sq[i, :]
            dist_sq[i, i] = np.inf
            
            new_min_pair_sq = np.min(dist_sq)
            new_min_wall = np.min(wall)
            new_score = min(np.sqrt(new_min_pair_sq) / 2.0, new_min_wall)
            
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_centers = centers.copy()
            else:
                # Revert
                centers[i] = old_pos
                wall[i] = old_wall_i
                diff_i = centers - old_pos
                dist_sq[i, :] = np.sum(diff_i ** 2, axis=1)
                dist_sq[:, i] = dist_sq[i, :]
                dist_sq[i, i] = np.inf
                no_improve += 1
        else:
            # Bad move - accept with probability exp(delta/T)
            delta = upper_bound - cur_score
            if delta / T > -40 and rng.random() < np.exp(delta / T):
                centers[i] = new_pos
                wall[i] = new_wall_i
                
                diff_i = centers - new_pos
                dist_sq[i, :] = np.sum(diff_i ** 2, axis=1)
                dist_sq[:, i] = dist_sq[i, :]
                dist_sq[i, i] = np.inf
                
                new_min_pair_sq = np.min(dist_sq)
                new_min_wall = np.min(wall)
                cur_score = min(np.sqrt(new_min_pair_sq) / 2.0, new_min_wall)
                no_improve += 1
            else:
                no_improve += 1
        
        # Adaptive step size
        if no_improve > max_no_improve:
            step *= 0.7
            no_improve = 0
        elif it % 5000 == 0 and it > 0:
            step *= 0.995
        
        # Restart exploration periodically
        if it % 50000 == 0 and it > 0:
            step = base_step * (0.5 + 1.0 * rng.random())
            no_improve = 0
    
    # Final local refinement
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5.0:
        centers = best_centers.copy()
        cur_score = _compute_score(centers)
        refine_step = 0.003
        refine_start = time.time()
        
        while time.time() - refine_start < remaining - 2.0:
            improved = False
            for _ in range(200):
                i = rng.integers(n)
                old_pos = centers[i].copy()
                new_pos = old_pos + rng.normal(0, refine_step, 2)
                new_pos = np.clip(new_pos, -0.4999, 0.4999)
                centers[i] = new_pos
                new_score = _compute_score(centers)
                
                if new_score > cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_centers = centers.copy()
                else:
                    centers[i] = old_pos
            
            if not improved:
                refine_step *= 0.5
                if refine_step < 1e-7:
                    break
    
    return best_centers, best_score


def _make_hex_init(rng, spacing=0.19):
    """Create hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    row_h = spacing * np.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < N and row < 15:
        offset = 0 if row % 2 == 0 else spacing / 2
        col = 0
        while idx < N and col < 12:
            x = -0.48 + offset + col * spacing
            y = -0.48 + row * row_h
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    if idx == N:
        centers += rng.normal(0, 0.002, (N, 2))
        centers = np.clip(centers, -0.4999, 0.4999)
    return centers


def _make_grid_init(rows, cols, rng):
    """Create grid initialization with jitter."""
    x_ticks = np.linspace(-0.48, 0.48, cols)
    y_ticks = np.linspace(-0.48, 0.48, rows)
    centers = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    centers += rng.normal(0, 0.003, (N, 2))
    centers = np.clip(centers, -0.4999, 0.4999)
    return centers


def construct():
    """
    Multi-start simulated annealing with diverse initializations.
    """
    rng = np.random.default_rng(42)
    start_time = time.time()
    max_time = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # Generate diverse initializations
    inits = []
    
    # Grid configurations
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]:
        inits.append(_make_grid_init(rows, cols, rng))
    
    # Hexagonal with different spacings
    for sp in [0.18, 0.19, 0.20]:
        inits.append(_make_hex_init(rng, sp))
    
    # Random
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # Run SA from each initialization
    for init_idx, init_c in enumerate(inits):
        remaining = max_time - (time.time() - start_time)
        if remaining < 8.0:
            break
        
        n_remaining = len(inits) - init_idx
        budget = min(remaining, max(12.0, remaining / n_remaining))
        
        c, s = _sa_run(init_c, rng, budget)
        
        if s > best_score:
            best_score = s
            best_centers = c
    
    # Fallback
    if best_centers is None:
        best_centers = _make_grid_init(7, 8, rng)
    
    return best_centers
# EVOLVE-BLOCK-END
