import numpy as np

N = 200

def construct():
    # 1. Generate initial hexagonal grids for various row counts
    best_init = None
    best_init_score = -1
    
    # Try different numbers of rows
    for m in range(10, 22):
        # Distribute N points into m rows as evenly as possible
        base = N // m
        rem = N % m
        # First 'rem' rows have base+1 points, rest have base
        row_counts = [base + 1] * rem + [base] * (m - rem)
        
        # Calculate the optimal dx for this configuration
        n_max = max(row_counts)
        # Constraint 1: Width
        # (n_max - 1) * dx + 2 * (dx/2) <= 1  => n_max * dx <= 1
        dx_width = 1.0 / n_max
        
        # Constraint 2: Height
        # (m - 1) * dy + 2 * (dx/2) <= 1
        # dy = dx * sqrt(3) / 2
        # (m - 1) * dx * sqrt(3) / 2 + dx <= 1
        # dx * (1 + (m - 1) * sqrt(3) / 2) <= 1
        dx_height = 1.0 / (1.0 + (m - 1) * np.sqrt(3) / 2.0)
        
        dx = min(dx_width, dx_height)
        dy = dx * np.sqrt(3) / 2.0
        r_est = dx / 2.0
        
        if r_est > best_init_score:
            best_init_score = r_est
            best_init = (m, row_counts, dx, dy)
            
    m, row_counts, dx, dy = best_init
    
    # Generate the grid
    centers = []
    y_start = -((m - 1) * dy) / 2.0
    for j, n in enumerate(row_counts):
        y = y_start + j * dy
        x_start = -((n - 1) * dx) / 2.0
        for i in range(n):
            x = x_start + i * dx
            centers.append([x, y])
            
    P = np.array(centers, dtype=float)
    
    # 2. Local Optimization
    # We will perform a local search to improve the minimum distance.
    # Score: min(min_pair_dist, min_bound_dist)
    
    def get_score(P):
        # Pairwise distances
        # P shape (N, 2)
        # Using broadcasting
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        # dist to left: 0.5 - P[:, 0]
        # dist to right: 0.5 + P[:, 0]
        # dist to bottom: 0.5 - P[:, 1]
        # dist to top: 0.5 + P[:, 1]
        bound_dists = np.min([
            0.5 - P[:, 0],
            0.5 + P[:, 0],
            0.5 - P[:, 1],
            0.5 + P[:, 1]
        ], axis=0)
        min_bound = np.min(bound_dists)
        
        return min(min_pair, min_bound)
    
    current_score = get_score(P)
    
    # Optimization Loop
    # Time limit is 120s. Let's run for a fixed number of iterations or time.
    # N=200, get_score is O(N^2) ~ 40k ops.
    # 10,000 iterations * 40k ops = 400M ops. In NumPy, this is fast (< 1-2s).
    # We can do more. Let's do 50,000 iterations.
    
    import time
    start_time = time.time()
    max_time = 100.0 # Leave buffer for other overheads
    
    for it in range(100000):
        if time.time() - start_time > max_time:
            break
            
        # Identify bottleneck
        # Recompute details to find the min pair and min bound
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_bound_vals = np.min([
            0.5 - P[:, 0],
            0.5 + P[:, 0],
            0.5 - P[:, 1],
            0.5 + P[:, 1]
        ], axis=0)
        min_bound_val = np.min(min_bound_vals)
        
        if min_pair_val < min_bound_val:
            # Bottleneck is a pair
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            if i == j: continue
            # Try to move i away from j or j away from i
            vec = P[i] - P[j]
            norm = np.linalg.norm(vec)
            if norm > 0:
                dir_vec = vec / norm
            else:
                dir_vec = np.random.randn(2)
                dir_vec /= np.linalg.norm(dir_vec)
                
            # Try several perturbations
            for _ in range(10):
                # Random direction with bias towards separating
                random_dir = np.random.randn(2)
                move_dir = 0.5 * dir_vec + 0.5 * random_dir
                move_dir /= np.linalg.norm(move_dir)
                
                step = 0.001 * np.random.uniform(0.5, 2.0)
                
                # Try moving i
                P_old_i = P[i].copy()
                P[i] = P_old_i + move_dir * step
                new_score = get_score(P)
                if new_score > current_score:
                    current_score = new_score
                    break
                else:
                    P[i] = P_old_i
                    
                # Try moving j in opposite direction
                P_old_j = P[j].copy()
                P[j] = P_old_j - move_dir * step
                new_score = get_score(P)
                if new_score > current_score:
                    current_score = new_score
                    break
                else:
                    P[j] = P_old_j
        else:
            # Bottleneck is boundary
            idx = np.argmin(min_bound_vals)
            i = idx
            # Determine which boundary
            # 0: left, 1: right, 2: bottom, 3: top
            # Re-evaluate which specific bound is min for this point
            b_left = 0.5 - P[i, 0]
            b_right = 0.5 + P[i, 0]
            b_bottom = 0.5 - P[i, 1]
            b_top = 0.5 + P[i, 1]
            
            if b_left == min_bound_vals[i]:
                dir_vec = np.array([1, 0])
            elif b_right == min_bound_vals[i]:
                dir_vec = np.array([-1, 0])
            elif b_bottom == min_bound_vals[i]:
                dir_vec = np.array([0, 1])
            else:
                dir_vec = np.array([0, -1])
                
            for _ in range(10):
                random_dir = np.random.randn(2)
                move_dir = 0.5 * dir_vec + 0.5 * random_dir
                move_dir /= np.linalg.norm(move_dir)
                
                step = 0.001 * np.random.uniform(0.5, 2.0)
                
                P_old_i = P[i].copy()
                P[i] = P_old_i + move_dir * step
                new_score = get_score(P)
                if new_score > current_score:
                    current_score = new_score
                    break
                else:
                    P[i] = P_old_i
                    
    return P
