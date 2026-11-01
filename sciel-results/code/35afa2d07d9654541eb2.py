import numpy as np
import time
import random

N = 200

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    best_pos = None
    best_r = 0.0
    
    def compute_score(pos):
        # Efficient score calculation
        # 1. Wall constraints
        max_abs = np.max(np.abs(pos), axis=1)
        wall_dists = 0.5 - max_abs
        min_wall = np.min(wall_dists)
        
        # 2. Pairwise constraints
        # For N=200, O(N^2) is 40k ops, which is fast in numpy
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_pair / 2.0, min_wall)
        return r, min_pair, min_wall, dists

    def pairwise_repulsion(pos, dists, min_pair, min_wall, target_r, iterations=2000):
        """
        Deterministic local optimization:
        1. If min_pair < 2*min_wall, push closest pair apart.
        2. If min_wall < min_pair/2, push wall-violating points inward.
        """
        pos = pos.copy()
        dists = dists.copy()
        
        for _ in range(iterations):
            if time.time() - start_time > time_limit - 5:
                break
                
            current_r = min(min_pair / 2.0, min_wall)
            
            # Identify constraint type
            if min_pair < 2.0 * min_wall:
                # Pairwise violation is tighter
                # Find closest pair
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                i, j = idx
                
                vec = pos[i] - pos[j]
                norm = np.linalg.norm(vec)
                if norm < 1e-10:
                    # Random direction if coincident
                    angle = random.uniform(0, 2*np.pi)
                    vec = np.array([np.cos(angle), np.sin(angle)])
                    norm = 1.0
                else:
                    vec /= norm
                
                # Move apart
                # We want to increase dist to at least 2*target_r or just increase it
                # Simple step: move 50% of the way to satisfy 2*target_r if possible, 
                # or a fixed small step if target_r is not known/feasible yet.
                # Since target_r is the current best found, we just try to increase min_pair.
                # A small fixed step is safer to avoid oscillation.
                step = 0.001 
                # If they are very close, take a larger step
                if norm < 0.01:
                    step = 0.005
                if norm < 0.001:
                    step = 0.01
                    
                pos[i] += vec * step
                pos[j] -= vec * step
                
                # Update dists for i and j
                new_dists_i = np.linalg.norm(pos[i] - pos, axis=1)
                dists[i, :] = new_dists_i
                dists[:, i] = new_dists_i
                dists[i, i] = np.inf
                
                new_dists_j = np.linalg.norm(pos[j] - pos, axis=1)
                dists[j, :] = new_dists_j
                dists[:, j] = new_dists_j
                dists[j, j] = np.inf
                
                # Update min_pair
                # Instead of full min, check if the moved points created a new min
                # But since we moved them apart, the old min_pair (i,j) increased.
                # However, i might be closer to k, or j to l.
                # For simplicity and safety with N=200, recompute min_pair from D
                # But that's O(N^2). 
                # Optimization: min_pair = min(min_pair_new_i, min_pair_new_j, min_pair_old_excluding_i_j)
                # This is complex. Given N=200, full recompute is ~40k ops. 
                # 2000 iterations * 40k = 80M ops. Might be too slow in Python loop?
                # Let's check: 80M numpy operations is fast, but the loop overhead is high.
                # Let's try to be smarter.
                
                # Actually, let's just recompute min_pair from the updated rows/cols 
                # and the previous min.
                # The new global min is min(
                #   min(dists[i, :]), 
                #   min(dists[j, :]), 
                #   min(dists excluding i and j)
                # )
                # We don't have "dists excluding i and j" efficiently without a heap or full scan.
                # Full scan is O(N^2). 
                # Let's assume the bottleneck is the Python loop.
                # Let's reduce iterations or make the step adaptive.
                
                min_pair = np.min(dists)
                
            else:
                # Wall violation is tighter
                # Find point closest to wall
                idx = np.argmin(min_wall)
                i = idx
                
                # Move inward
                # Direction: towards center (0,0)
                vec = -pos[i]
                norm = np.linalg.norm(vec)
                if norm < 1e-10:
                    # If at center, move randomly? But if at center, wall dist is 0.5.
                    # If min_wall < min_pair/2, and pos is at center, then 0.5 < min_pair/2 -> min_pair > 1.
                    # Impossible for N=200 in unit square.
                    # So pos[i] is not at center.
                    vec = np.array([1.0, 0.0])
                    norm = 1.0
                else:
                    vec /= norm
                    
                step = 0.001
                pos[i] += vec * step
                
                # Update dists for i
                new_dists_i = np.linalg.norm(pos[i] - pos, axis=1)
                dists[i, :] = new_dists_i
                dists[:, i] = new_dists_i
                dists[i, i] = np.inf
                
                # Update wall dists
                max_abs = np.max(np.abs(pos[i]))
                min_wall = 0.5 - max_abs
                # Need to recompute min_wall for all?
                # Only i changed.
                # min_wall = min(min_wall_old_excluding_i, new_wall_i)
                # Again, full recompute is O(N).
                wall_dists = 0.5 - np.max(np.abs(pos), axis=1)
                min_wall = np.min(wall_dists)
                
                min_pair = np.min(dists)
                
        return pos, dists, min_pair, min_wall

    # Initialization strategies
    def init_grid():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pos = grid[:N].copy()
        pos += np.random.normal(0, 0.001, pos.shape)
        return pos

    def init_hex():
        # Hexagonal packing approximation
        # Spacing s = 1/sqrt(3) ~ 0.577 for infinite plane.
        # For N=200, we need to fit in 1x1.
        # Area per circle ~ 1/200 = 0.005.
        # Hex area per circle ~ (sqrt(3)/2) * s^2.
        # s^2 ~ 0.005 * 2 / 1.732 ~ 0.0057 -> s ~ 0.075
        # This is just a guess. Let's use a grid with offset rows.
        pos = []
        rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
        cols = int(np.ceil(np.sqrt(N * np.sqrt(3) / 2)))
        
        # Generate hex grid points
        # x = col * s + (row % 2) * s/2
        # y = row * s * sqrt(3)/2
        s = 0.08 # Guess
        points = []
        for r in range(rows):
            for c in range(cols):
                x = c * s + (r % 2) * s * 0.5
                y = r * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        # Center and scale to fit in [-0.5, 0.5]
        # Find bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = (min_coords + max_coords) / 2
        points -= center
        
        # Scale to fit
        max_range = np.max(np.abs(points))
        if max_range > 0:
            points *= 0.45 / max_range # Leave margin
            
        # Shuffle and take N
        idx = np.random.choice(len(points), N, replace=False)
        pos = points[idx]
        pos += np.random.normal(0, 0.001, pos.shape)
        return pos

    def init_random():
        pos = np.random.uniform(-0.45, 0.45, (N, 2))
        return pos

    initializers = [init_grid, init_hex, init_random]
    
    # Run multiple restarts
    for init_idx, init_func in enumerate(initializers):
        if time.time() - start_time > time_limit - 10:
            break
            
        pos = init_func()
        # Ensure valid initial position (inside square)
        pos = np.clip(pos, -0.49, 0.49)
        
        r, min_pair, min_wall, dists = compute_score(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            
        # Local optimization
        # Use a fixed number of iterations or time-based
        # Let's do 500 iterations per restart
        pos, dists, min_pair, min_wall = pairwise_repulsion(pos, dists, min_pair, min_wall, best_r, iterations=500)
        
        r, _, _, _ = compute_score(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            
        # If we have time, do another pass with more iterations
        if time.time() - start_time < time_limit - 20:
            pos, dists, min_pair, min_wall = pairwise_repulsion(pos, dists, min_pair, min_wall, best_r, iterations=1000)
            r, _, _, _ = compute_score(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()

    # Final validation and return
    if best_pos is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # Clip to ensure validity
    best_pos = np.clip(best_pos, -0.5, 0.5)
    return best_pos

# Keep the function name exactly as construct
