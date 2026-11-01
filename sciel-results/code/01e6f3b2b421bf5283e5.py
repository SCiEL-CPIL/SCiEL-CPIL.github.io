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


def _sa_run(centers, rng, time_budget):
    """
    Run simulated annealing on centers with proper incremental scoring.
    Returns (best_centers, best_score).
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Precompute initial state
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 2)
    current_score = _full_score(centers)
    
    # Track the global minimum pair distance and which pair achieves it
    # For efficiency, we'll recompute the full score when needed
    # But we can be smarter: track min_pair_sq and min_wall separately
    
    best_centers = centers.copy()
    best_score = current_score
    
    # SA parameters
    n_iters = 800000
    T_start = 0.015
    T_end = 1e-9
    alpha = (T_end / T_start) ** (1.0 / n_iters)
    
    # Adaptive step size
    step = 0.02
    base_step = 0.02
    no_improve = 0
    max_no_improve = 2000
    
    # For incremental scoring, we need to track:
    # - min_wall: the minimum wall distance across all circles
    # - min_pair_sq: the minimum squared pair distance
    # When we move circle i, only distances involving i change
    
    # Let's use a different strategy: maintain the full distance matrix
    # and update it incrementally when a circle moves
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    current_min_pair_sq = np.min(dists_sq)
    current_min_wall = np.min(wall_dists)
    current_score = min(np.sqrt(current_min_pair_sq) / 2.0, current_min_wall)
    
    best_score = current_score
    best_centers = centers.copy()
    
    for it in range(n_iters):
        # Time check every 500 iterations
        if it % 500 == 0:
            if time.time() - start_time > time_budget:
                break
        
        T = T_start * (alpha ** it)
        
        # Pick random circle to move
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Generate new position
        new_pos = old_pos + rng.normal(0, step, 2)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Compute new distances from i to all others
        new_dists_sq_i = np.sum((centers - new_pos) ** 2, axis=1)
        new_dists_sq_i[i] = np.inf
        
        # New wall distance for circle i
        new_wall_i = np.min(0.5 - np.abs(new_pos))
        
        # To compute the new global score, we need:
        # 1. New min wall = min(min of wall_dists excluding i's old, new_wall_i)
        # 2. New min pair = min(min of dists_sq excluding row/col i, new_dists_sq_i)
        
        # Efficient approach: 
        # - For wall: if old wall_i was the unique min, we need the second min
        # - For pair: if old pair involved i, we need the second min pair
        
        # Simpler but correct: compute the new min wall and new min pair
        # New min wall: min of all wall distances with i updated
        # We can compute this as min(new_wall_i, min of wall_dists for j != i)
        
        # For wall: 
        old_wall_i = np.min(0.5 - np.abs(old_pos))
        # Find min wall excluding i
        wall_excl_i = np.concatenate([wall_dists[:i], wall_dists[i+1:]]).min() if N > 1 else np.inf
        new_min_wall = min(new_wall_i, wall_excl_i)
        
        # For pair:
        # The old min pair might involve i or not
        # New min pair = min(min pair not involving i, new_dists_sq_i)
        # To get min pair not involving i, we need to exclude row and col i
        
        # This is getting complex. Let's use a hybrid approach:
        # Compute the candidate new score, and if it's clearly better, accept
        # If it's close, do a full recompute
        
        # Quick estimate: the new score is at most min(new_min_pair_involving_i/2, new_min_wall)
        # where new_min_pair_involving_i = min(new_dists_sq_i)
        # But the actual new score could be limited by pairs not involving i
        
        # Let's compute the new min pair properly:
        # Create a temporary dists_sq with row/col i removed
        # This is expensive. Instead, let's track the top-2 min pairs
        
        # Alternative: just do a full recompute every K iterations, 
        # and use incremental for the rest with a conservative estimate
        
        # For now, let's use the following strategy:
        # Compute new score as min(new_min_pair_involving_i / 2, new_min_wall)
        # This is an upper bound on the true new score (true score <= this)
        # If this upper bound > current_score, the move might be good
        # If this upper bound < current_score, the move is definitely bad
        
        new_min_pair_inv_i = np.min(new_dists_sq_i)
        candidate_score = min(np.sqrt(new_min_pair_inv_i) / 2.0, new_min_wall)
        
        # If candidate_score > current_score, the move could improve things
        # We need to verify with a full computation
        # If candidate_score <= current_score, the move cannot improve (since true_score <= candidate_score)
        
        if candidate_score > current_score:
            # Apply the move and do full recompute
            centers[i] = new_pos
            wall_dists[i] = 0.5 - np.abs(new_pos)
            
            # Update dists_sq
            diff_i = centers - new_pos
            dists_sq[i, :] = np.sum(diff_i ** 2, axis=1)
            dists_sq[:, i] = dists_sq[i, :]
            dists_sq[i, i] = np.inf
            
            new_min_pair_sq = np.min(dists_sq)
            new_min_wall = np.min(wall_dists)
            new_score = min(np.sqrt(new_min_pair_sq) / 2.0, new_min_wall)
            
            if new_score >= current_score:
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_centers = centers.copy()
            else:
                # Revert
                centers[i] = old_pos
                wall_dists[i] = 0.5 - np.abs(old_pos)
                diff_i = centers - old_pos
                dists_sq[i, :] = np.sum(diff_i ** 2, axis=1)
                dists_sq[:, i] = dists_sq[i, :]
                dists_sq[i, i] = np.inf
                no_improve += 1
        else:
            # The move cannot improve the score (upper bound is already <= current)
            # But in SA, we might still accept bad moves to escape local minima
            # However, if the upper bound is much worse, reject
            delta_upper = candidate_score - current_score
            if delta_upper / T > -50 and rng.random() < np.exp(delta_upper / T):
                # Accept this bad move (exploration)
                centers[i] = new_pos
                wall_dists[i] = 0.5 - np.abs(new_pos)
                
                diff_i = centers - new_pos
                dists_sq[i, :] = np.sum(diff_i ** 2, axis=1)
                dists_sq[:, i] = dists_sq[i, :]
                dists_sq[i, i] = np.inf
                
                new_min_pair_sq = np.min(dists_sq)
                new_min_wall = np.min(wall_dists)
                current_score = min(np.sqrt(new_min_pair_sq) / 2.0, new_min_wall)
                no_improve += 1
            else:
                no_improve += 1
        
        # Adaptive step size
        if no_improve > max_no_improve:
            step *= 0.5
            no_improve = 0
        elif it % 1000 == 0 and it > 0:
            step *= 0.998
        
        # Periodically increase step to escape local minima
        if it % 20000 == 0 and it > 0:
            step = base_step * (0.7 + 0.6 * rng.random())
            no_improve = 0
    
    # Final local refinement
    if time.time() - start_time < time_budget:
        centers = best_centers.copy()
        refine_budget = min(15.0, time_budget - (time.time() - start_time))
        refine_start = time.time()
        refine_step = 0.005
        cur_score = _full_score(centers)
        
        # Greedy local search with decreasing step
        while time.time() - refine_start < refine_budget:
            improved = False
            for _ in range(500):
                i = rng.integers(N)
                old_pos = centers[i].copy()
                new_pos = old_pos + rng.normal(0, refine_step, 2)
                new_pos = np.clip(new_pos, -0.4999, 0.4999)
                centers[i] = new_pos
                new_score = _full_score(centers)
                
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
                if refine_step < 1e-6:
                    break
    
    return best_centers, best_score


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
    
    # Grid-based initializations with different aspect ratios
    grid_configs = [
        (7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (5, 5, 10),  # last is special
    ]
    
    for rows, cols in grid_configs[:6]:
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
        # Add small random jitter
        centers += rng.normal(0, 0.003, (N, 2))
        centers = np.clip(centers, -0.4999, 0.4999)
        initializations.append(centers)
    
    # Hexagonal-like packing
    hex_centers = np.zeros((N, 2))
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
                hex_centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    if idx == N:
        hex_centers += rng.normal(0, 0.003, (N, 2))
        hex_centers = np.clip(hex_centers, -0.4999, 0.4999)
        initializations.append(hex_centers)
    
    # Random initialization
    initializations.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # Run SA from each initialization
    for init_idx, init_centers in enumerate(initializations):
        remaining = max_time - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        # Allocate time budget proportionally, but ensure each gets at least 10s
        time_budget = min(remaining, max(10.0, remaining / (len(initializations) - init_idx)))
        
        centers_local, score_local = _sa_run(init_centers, rng, time_budget)
        
        if score_local > best_score:
            best_score = score_local
            best_centers = centers_local
    
    # Final verification
    if best_centers is None:
        # Fallback
        x_ticks = np.linspace(-0.48, 0.48, 8)
        y_ticks = np.linspace(-0.48, 0.48, 7)
        best_centers = np.zeros((N, 2))
        idx = 0
        for r in range(7):
            for c in range(8):
                if idx < N:
                    best_centers[idx] = [x_ticks[c], y_ticks[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
    
    return best_centers
# EVOLVE-BLOCK-END
