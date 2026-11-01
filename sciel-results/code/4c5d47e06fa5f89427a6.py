"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _full_score(centers):
    """Compute the packing radius: min(min_pair_dist/2, min_wall_dist)."""
    min_wall = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq))
    return min(min_pair / 2.0, min_wall)


def _compute_min_pair_and_wall(centers):
    """Compute min pair distance and min wall distance efficiently."""
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    return np.sqrt(min_pair_sq) / 2.0, min_wall


def _sa_run(centers, rng, time_budget):
    """
    Run simulated annealing with efficient incremental scoring.
    Uses tracking of top-2 minimums for fast updates.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Initial computation
    wall_dists = 0.5 - np.abs(centers)
    current_score = _full_score(centers)
    
    # Build distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Track the minimum pair distance and which pair achieves it
    # For efficiency, track the smallest pair distance
    current_min_pair_sq = np.min(dists_sq)
    current_min_wall = np.min(wall_dists)
    current_score = min(np.sqrt(current_min_pair_sq) / 2.0, current_min_wall)
    
    best_centers = centers.copy()
    best_score = current_score
    
    # SA parameters - geometric cooling with restarts
    T_start = 0.02
    T_min = 1e-6
    cooling_rate = 0.99998  # Slow cooling for better exploration
    step = 0.015
    base_step = 0.015
    
    # Track acceptance rate for adaptive step size
    accept_count = 0
    window_size = 500
    window_accepts = 0
    
    it = 0
    max_iters = 1000000
    
    while time.time() - start_time < time_budget and it < max_iters:
        # Time check
        if it % 200 == 0 and time.time() - start_time > time_budget:
            break
        
        T = max(T_start * (cooling_rate ** it), T_min)
        
        # Pick random circle to move
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Generate new position with adaptive step
        new_pos = old_pos + rng.normal(0, step, 2)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Compute new distances from i to all others
        delta = centers - new_pos
        new_dists_sq_i = np.sum(delta * delta, axis=1)
        new_dists_sq_i[i] = np.inf
        
        # New wall distance for circle i
        new_wall_i = np.min(0.5 - np.abs(new_pos))
        
        # Compute the new global score efficiently
        # We need: new_min_pair = min(min pair not involving i, min new_dists_sq_i)
        # and new_min_wall = min(min wall not involving i, new_wall_i)
        
        # For wall: find min wall excluding circle i
        if N > 1:
            # Temporarily set i's wall to inf and find min
            old_wall_i = wall_dists[i].copy()
            wall_dists[i] = np.inf
            min_wall_excl_i = np.min(wall_dists)
            wall_dists[i] = old_wall_i
        else:
            min_wall_excl_i = np.inf
        new_min_wall = min(new_wall_i, min_wall_excl_i)
        
        # For pair: we need the min pair distance not involving i
        # This is the min of dists_sq excluding row and column i
        # Efficient: create a mask or use the fact that we can find min of 
        # the submatrix excluding i
        if N > 2:
            # Create a copy with row/col i set to inf
            # More efficient: iterate or use np.where
            # For N=50, this is fast enough
            dists_sq_no_i = dists_sq.copy()
            dists_sq_no_i[i, :] = np.inf
            dists_sq_no_i[:, i] = np.inf
            min_pair_sq_excl_i = np.min(dists_sq_no_i)
        else:
            min_pair_sq_excl_i = np.inf
        
        new_min_pair_sq = min(np.min(new_dists_sq_i), min_pair_sq_excl_i)
        new_score = min(np.sqrt(new_min_pair_sq) / 2.0, new_min_wall)
        
        # Accept/reject decision
        delta_score = new_score - current_score
        
        if delta_score >= 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
            # Accept the move
            centers[i] = new_pos
            wall_dists[i] = 0.5 - np.abs(new_pos)
            
            # Update distance matrix row/col i
            diff_i = centers - new_pos
            dists_sq[i, :] = np.sum(diff_i * diff_i, axis=1)
            dists_sq[:, i] = dists_sq[i, :]
            dists_sq[i, i] = np.inf
            
            current_score = new_score
            accept_count += 1
            window_accepts += 1
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        
        it += 1
        
        # Adaptive step size based on acceptance rate
        if it % window_size == 0:
            acceptance_rate = window_accepts / window_size
            window_accepts = 0
            
            # Target acceptance rate ~ 20-40%
            if acceptance_rate < 0.1:
                step *= 0.8
            elif acceptance_rate > 0.5:
                step *= 1.2
            else:
                step *= 0.995
            
            step = np.clip(step, 0.001, 0.05)
        
        # Periodic restart to escape local minima
        if it % 50000 == 0 and it > 0:
            # Slightly perturb all positions
            centers += rng.normal(0, 0.002, (N, 2))
            centers = np.clip(centers, -0.4999, 0.4999)
            
            # Recompute distance matrix
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            wall_dists = 0.5 - np.abs(centers)
            
            current_score = min(np.sqrt(np.min(dists_sq)) / 2.0, np.min(wall_dists))
            step = base_step * (0.8 + 0.4 * rng.random())
        
        # Reset temperature occasionally for exploration
        if it % 100000 == 0 and it > 0:
            T_start = max(T_start * 0.5, 0.001)
    
    # Final local refinement using coordinate descent
    if time.time() - start_time < time_budget:
        centers = best_centers.copy()
        refine_budget = min(20.0, time_budget - (time.time() - start_time))
        refine_start = time.time()
        cur_score = _full_score(centers)
        
        # Coordinate descent: optimize each coordinate separately
        refine_step = 0.002
        while time.time() - refine_start < refine_budget:
            improved = False
            for coord_iter in range(3):
                for i in range(N):
                    for dim in range(2):
                        old_val = centers[i, dim]
                        # Try small moves in each direction
                        for delta in [refine_step, -refine_step, refine_step * 0.5, -refine_step * 0.5]:
                            new_val = old_val + delta
                            if -0.4999 <= new_val <= 0.4999:
                                centers[i, dim] = new_val
                                new_score = _full_score(centers)
                                if new_score > cur_score:
                                    cur_score = new_score
                                    improved = True
                                    if cur_score > best_score:
                                        best_score = cur_score
                                        best_centers = centers.copy()
                                    break
                        else:
                            centers[i, dim] = old_val
                
                if not improved:
                    break
            
            if not improved:
                refine_step *= 0.5
                if refine_step < 1e-7:
                    break
            else:
                improved = False
    
    return best_centers, best_score


def _make_grid_init(rows, cols, rng, jitter=0.005):
    """Create a grid initialization with jitter."""
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
    if idx < N:
        # Fill remaining with random positions
        centers[idx:] = rng.uniform(-0.45, 0.45, (N - idx, 2))
    centers += rng.normal(0, jitter, (N, 2))
    return np.clip(centers, -0.4999, 0.4999)


def _make_hex_init(rng, jitter=0.005):
    """Create a hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    hex_spacing = 0.19
    hex_row_h = hex_spacing * np.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < N and row < 12:
        offset = 0 if row % 2 == 0 else hex_spacing / 2
        col = 0
        while idx < N and col < 10:
            x = -0.48 + offset + col * hex_spacing
            y = -0.48 + row * hex_row_h
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    if idx < N:
        centers[idx:] = rng.uniform(-0.45, 0.45, (N - idx, 2))
    centers += rng.normal(0, jitter, (N, 2))
    return np.clip(centers, -0.4999, 0.4999)


def construct():
    """
    Multi-start simulated annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    max_time = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # Various grid configurations
    grid_configs = [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (5, 10), (10, 5)]
    for rows, cols in grid_configs:
        initializations.append(_make_grid_init(rows, cols, rng, jitter=0.004))
    
    # Hexagonal
    initializations.append(_make_hex_init(rng, jitter=0.004))
    
    # Random
    initializations.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # Run SA from each initialization
    n_inits = len(initializations)
    for init_idx, init_centers in enumerate(initializations):
        remaining = max_time - (time.time() - start_time)
        if remaining < 8.0:
            break
        
        # Allocate time: give more time to earlier inits, ensure minimum
        if init_idx == 0:
            time_budget = min(remaining, max(25.0, remaining * 0.35))
        elif init_idx == n_inits - 1:
            time_budget = min(remaining, max(8.0, remaining / 2))
        else:
            time_budget = min(remaining, max(12.0, remaining / (n_inits - init_idx)))
        
        centers_local, score_local = _sa_run(init_centers, rng, time_budget)
        
        if score_local > best_score:
            best_score = score_local
            best_centers = centers_local
    
    # Final verification and fallback
    if best_centers is None:
        best_centers = _make_grid_init(7, 8, rng, jitter=0.003)
    
    return best_centers
# EVOLVE-BLOCK-END
