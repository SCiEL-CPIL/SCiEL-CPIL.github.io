"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_min_dist_and_wall(centers):
    """
    Compute the minimum pairwise distance and minimum wall distance.
    Returns (min_pair_dist, min_wall_dist).
    """
    # Wall distances
    min_wall = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances using broadcasting (fast enough for N=50 if done sparingly, 
    # but we will use a neighbor list approach in the main loop for speed)
    # For initial setup or verification, we can do it fully.
    diff = centers[:, np.newaxis] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min_pair, min_wall


def _update_local_scores(i, centers, min_dists, min_wall_dists, min_pair, min_wall):
    """
    Update the minimum distances involving center i.
    This is an optimization: instead of recomputing all N^2 distances,
    we only update the distances from i to all others.
    
    Note: This still requires O(N) work per move, which is much better than O(N^2).
    """
    # Update wall distance for i
    new_wall_i = 0.5 - np.max(np.abs(centers[i]))
    
    # Update pair distances involving i
    # dists[i, j] = |centers[i] - centers[j]|
    diffs = centers[i] - centers
    dists_i = np.sqrt(np.sum(diffs * diffs, axis=1))
    dists_i[i] = np.inf # ignore self
    
    # The global min_pair is min(min_pair, min(dists_i))
    # But we need to be careful: if the old min_pair was determined by a pair NOT involving i,
    # and we moved i, the old min_pair might still be valid if it's smaller than any new dist_i.
    # However, if the old min_pair WAS involving i, and we moved i, that specific distance changed.
    # We don't track WHICH pair was the minimum, so we have to recompute the global min.
    # 
    # Optimization Strategy:
    # Maintain a list of the K smallest distances? No, too complex.
    # Since N=50 is small, O(N) per step is very fast.
    # Let's just compute the new global min by checking:
    # 1. Is the new min_wall_i smaller than current min_wall?
    # 2. Is any new dist_i smaller than current min_pair?
    # 3. BUT, if the old min_pair was between i and some j, and we moved i, 
    #    the old min_pair value is no longer necessarily a lower bound for the new min_pair.
    #    Actually, the new min_pair is min(old_min_pair_excluding_i, min_dist_i).
    #    We don't know old_min_pair_excluding_i.
    #    
    #    Alternative: Just recompute min_pair from scratch? O(N^2) = 2500 ops.
    #    In Python, even 2500 ops * 100,000 iters = 250M ops, which is ~5-10 seconds.
    #    This is acceptable.
    #    
    #    Let's stick to the full recomputation for simplicity and correctness, 
    #    but vectorized. The previous code did this.
    #    Why was it slow? Maybe 100k iters wasn't enough.
    #    Let's try to speed up the inner loop.
    
    return new_wall_i, dists_i


def construct():
    """
    Use simulated annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    best_overall = None
    best_overall_score = -np.inf
    
    # Time limit check
    start_time = time.time()
    max_time = 110.0 # Leave 10s margin
    
    # Initialization: Grid is usually a good start
    # 7x8 grid
    rows, cols = 7, 8
    x_ticks = np.linspace(-0.45, 0.45, cols)
    y_ticks = np.linspace(-0.45, 0.45, rows)
    
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
    
    # Add small jitter to break symmetry
    centers += rng.normal(0, 0.01, (N, 2))
    centers = np.clip(centers, -0.4999, 0.4999)
    
    # Precompute initial score
    min_pair, min_wall = _compute_min_dist_and_wall(centers)
    current_score = min(min_pair / 2.0, min_wall)
    
    best_centers = centers.copy()
    best_score = current_score
    
    n_iterations = 2000000
    T_start = 0.05
    T_end = 1e-12
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    base_step = 0.05
    step_size = base_step
    
    no_improve_count = 0
    max_no_improve = 10000
    
    for it in range(n_iterations):
        # Check time
        if it % 10000 == 0:
            if time.time() - start_time > max_time:
                break
        
        T = T_start * (alpha ** it)
        
        # Pick a random center
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Perturb
        centers[i] += rng.normal(0, step_size, 2)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        # Compute new score
        # To save time, we can check if the move is likely to improve.
        # But we need the exact score for SA acceptance.
        # Vectorized distance calculation for N=50 is fast.
        
        # Wall distance
        new_wall = np.min(0.5 - np.abs(centers))
        
        # Pair distance
        # Only compute distances involving i? No, we need global min.
        # However, if the new wall distance is already worse than current_score, 
        # and current_score was limited by wall, we might reject early?
        # Let's just compute it.
        
        diff = centers[:, np.newaxis] - centers[np.newaxis, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        new_pair = np.min(dists)
        
        new_score = min(new_pair / 2.0, new_wall)
        
        if new_score >= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            delta = new_score - current_score
            # Accept with prob exp(delta/T)
            if delta / T > -100:
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    no_improve_count = 0
                else:
                    centers[i] = old_pos
                    no_improve_count += 1
            else:
                centers[i] = old_pos
                no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > max_no_improve:
            step_size *= 0.5
            no_improve_count = 0
        elif it % 5000 == 0 and it > 0:
            step_size *= 0.98
            
        # Periodically restart step size to escape local minima
        if it % 50000 == 0 and it > 0:
            step_size = base_step * (0.5 + 0.5 * rng.random())
            no_improve_count = 0
            
    return best_centers
# EVOLVE-BLOCK-END
