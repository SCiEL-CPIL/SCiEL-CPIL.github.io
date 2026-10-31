import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def initialize_hexagonal():
        """
        Initialize with a hexagonal-like grid to maximize initial packing density.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Arrange in a grid, slightly offset for hexagonal packing
        # 5x5 grid is 25, so we need 26. Let's do a 5x5 + 1 or just randomize a grid.
        # Better: Just use a simple grid with random jitter to break symmetry.
        side = 5
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    # Jitter to avoid perfect symmetry which causes issues with ties
                    x[count] = (i + 0.5) / side + np.random.uniform(-0.02, 0.02)
                    y[count] = (j + 0.5) / side + np.random.uniform(-0.02, 0.02)
                    # Initial radius: small enough to not overlap neighbors in grid
                    # Grid spacing is 1/5 = 0.2. Max radius is 0.1.
                    r[count] = 0.09
                    count += 1
        
        # Clip to bounds
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        return x, y, r

    def initialize_random():
        x = np.random.rand(N) * 0.8 + 0.1
        y = np.random.rand(N) * 0.8 + 0.1
        r = np.full(N, 0.02)
        return x, y, r

    def get_max_possible_radius(x, y, r, idx):
        """
        Calculate the maximum radius circle 'idx' can have, given current positions of all circles.
        It is limited by:
        1. Distance to boundaries.
        2. Distance to other circles (dist / 2 if we could move others, but we can't).
           Actually, since positions are fixed, the max radius is:
           min(
               min(x_i, 1-x_i, y_i, 1-y_i),
               min_{j != i} (dist(i,j) - r_j)  # Wait, this is only valid if we assume others stay fixed.
           )
           But if we increase r_i, it might overlap j. The constraint is r_i + r_j <= dist(i,j).
           So r_i <= dist(i,j) - r_j.
        """
        # Boundary limits
        max_r_bound = min(x[idx], 1.0 - x[idx], y[idx], 1.0 - y[idx])
        
        # Neighbor limits
        max_r_neighbor = max_r_bound
        for j in range(N):
            if j == idx:
                continue
            dx = x[idx] - x[j]
            dy = y[idx] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            # r_i + r_j <= dist  =>  r_i <= dist - r_j
            limit = dist - r[j]
            if limit < max_r_neighbor:
                max_r_neighbor = limit
        
        return max(0.0, max_r_neighbor)

    def grow_all(x, y, r, steps=1000, step_size=0.001):
        """
        Iteratively try to grow each circle.
        If a circle can grow, grow it by a small amount.
        This is a greedy approach.
        """
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        
        for _ in range(steps):
            progress = False
            for idx in indices:
                # Check if we can grow
                current_r = r[idx]
                max_r = get_max_possible_radius(x, y, r, idx)
                
                if max_r > current_r + 1e-6:
                    # Grow towards max_r, but limited by step_size to allow others to adjust?
                    # No, if we grow to max_r, it might block others.
                    # But growing is always good for sum(r) unless it prevents a larger growth elsewhere.
                    # Greedy: grow as much as possible? Or small steps?
                    # Small steps allow "negotiation" if we were moving positions.
                    # Since positions are fixed, growing to max is optimal for this state.
                    # However, growing one circle might reduce the max radius of others (by reducing their available space? No, positions are fixed).
                    # Wait. If positions are fixed, r_i and r_j are independent constraints?
                    # Constraint: r_i + r_j <= d_ij.
                    # If I increase r_i, the max r_j decreases.
                    # So there is a trade-off.
                    # Greedy growth might get stuck in a suboptimal distribution (e.g., one huge circle, many tiny).
                    # To handle this, we can grow randomly or use a more sophisticated allocation.
                    
                    # For now, let's grow by a small amount to allow multiple circles to grow in one pass.
                    growth = min(step_size, max_r - current_r)
                    r[idx] += growth
                    progress = True
            
            if not progress:
                break
                
        return x, y, r

    def shuffle_positions(x, y, r, strength=0.05):
        """
        Randomly perturb positions to escape local minima.
        Must maintain validity (non-overlap).
        This is tricky. We can only move a circle if it doesn't cause overlap.
        Or, we can move it, check validity, and revert if invalid.
        """
        x_new = x.copy()
        y_new = y.copy()
        
        # Try to move each circle
        indices = np.random.permutation(N)
        for idx in indices:
            # Random direction and magnitude
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(0, strength)
            
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            new_xi = x_new[idx] + dx
            new_yi = y_new[idx] + dy
            
            # Check boundaries
            if new_xi < r[idx] or new_xi > 1 - r[idx] or new_yi < r[idx] or new_yi > 1 - r[idx]:
                continue
                
            # Check neighbors
            valid = True
            for j in range(N):
                if j == idx:
                    continue
                dxn = new_xi - x_new[j]
                dyn = new_yi - y_new[j]
                dist_sq = dxn*dxn + dyn*dyn
                sum_r = r[idx] + r[j]
                if dist_sq < sum_r*sum_r - 1e-9:
                    valid = False
                    break
            
            if valid:
                x_new[idx] = new_xi
                y_new[idx] = new_yi
                
        return x_new, y_new

    def optimize_single_run(x, y, r, time_limit):
        """
        Main optimization loop for one starting configuration.
        """
        best_r_sum = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Initial growth
        x, y, r = grow_all(x, y, r, steps=500, step_size=0.01)
        if np.sum(r) > best_r_sum:
            best_r_sum = np.sum(r)
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
        while time.time() < time_limit:
            # 1. Grow
            x, y, r = grow_all(x, y, r, steps=100, step_size=0.005)
            if np.sum(r) > best_r_sum:
                best_r_sum = np.sum(r)
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                
            # 2. Shuffle to escape local minima
            # Strength of shuffle can be adapted: if stuck, shuffle harder?
            # For now, constant strength.
            x_new, y_new = shuffle_positions(x, y, r, strength=0.02)
            
            # After shuffle, we might have lost some "growth potential" if we moved away from boundaries/neighbors?
            # No, moving is neutral for sum(r) initially, but changes the constraint landscape.
            # We must re-grow after shuffle.
            
            # Check if shuffle actually helped by doing a quick grow
            x_temp, y_temp, r_temp = x_new, y_new, r.copy()
            x_temp, y_temp, r_temp = grow_all(x_temp, y_temp, r_temp, steps=50, step_size=0.01)
            
            if np.sum(r_temp) > np.sum(r):
                x, y, r = x_temp, y_temp, r_temp
            else:
                # Revert shuffle if it didn't help immediately? 
                # Or keep it for diversity?
                # Let's keep it to explore, but if we are far from best, maybe revert?
                # Simple strategy: Keep it. The next grow might find a better spot.
                x, y = x_new, y_new
                
            # 3. Refine: Try to grow more aggressively
            x, y, r = grow_all(x, y, r, steps=200, step_size=0.002)
            
            if np.sum(r) > best_r_sum:
                best_r_sum = np.sum(r)
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                
        return best_x, best_y, best_r

    best_x_global = None
    best_y_global = None
    best_r_global = None
    best_score_global = 0.0
    
    # Multi-start
    num_starts = 5
    time_per_start = (deadline - start_time) / num_starts
    
    for k in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        start_k = time.time()
        limit_k = start_k + time_per_start - 1.0 # 1s buffer
        
        if k == 0:
            x, y, r = initialize_hexagonal()
        elif k == 1:
            x, y, r = initialize_random()
        else:
            # Mix: start from previous best but perturb?
            # Or just random
            x, y, r = initialize_random()
            # Add some structure?
            # Let's just use random for now, hex was for k=0.
            
        x_res, y_res, r_res = optimize_single_run(x, y, r, limit_k)
        
        score = np.sum(r_res)
        if score > best_score_global:
            best_score_global = score
            best_x_global = x_res
            best_y_global = y_res
            best_r_global = r_res
            
    # Fallback
    if best_x_global is None:
        best_x_global, best_y_global, best_r_global = initialize_hexagonal()
        best_x_global, best_y_global, best_r_global = grow_all(best_x_global, best_y_global, best_r_global, steps=1000, step_size=0.01)
        
    # Final Safety Check & Projection
    # Ensure strict validity
    x_f, y_f, r_f = best_x_global, best_y_global, best_r_global
    
    # Shrink if necessary (should not be needed if algorithm is correct, but just in case)
    for _ in range(100):
        valid = True
        # Boundaries
        for i in range(N):
            if x_f[i] < r_f[i] - 1e-9 or x_f[i] > 1 - r_f[i] + 1e-9 or y_f[i] < r_f[i] - 1e-9 or y_f[i] > 1 - r_f[i] + 1e-9:
                valid = False
                break
        if not valid:
            # Shrink all slightly
            r_f *= 0.98
            continue
            
        # Pairs
        for i in range(N):
            for j in range(i+1, N):
                dx = x_f[i] - x_f[j]
                dy = y_f[i] - y_f[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r_f[i] + r_f[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            break
        r_f *= 0.95
        
    final_state = np.column_stack([x_f, y_f, r_f])
    return final_state

# EVOLVE-BLOCK-END
