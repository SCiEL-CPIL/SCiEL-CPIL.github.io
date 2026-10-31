"""
Strategy: Greedy Sequential Circle Packing with Randomized Restarts
Instead of optimizing coordinates of all circles simultaneously, we build the packing
incrementally. For a given aspect ratio, we randomly select an insertion order.
For each circle in that order, we find the position and maximum radius that allows
it to fit without overlapping existing circles and staying within the rectangle.
This avoids local minima of coordinate optimizers and naturally respects non-overlap.
"""
import numpy as np
import time
import random
from scipy.spatial import cKDTree

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    
    def try_packing(W, H, order, time_limit):
        """
        Attempt to pack N circles in a rectangle W x H using the given insertion order.
        Returns (sum_r, config) or (0, None) if it fails or times out.
        """
        current_time = time.time()
        if current_time > time_limit:
            return 0.0, None
        
        placed = []  # List of (x, y, r)
        
        # We will place circles one by one.
        # To make it efficient, we maintain a list of placed circles.
        # For each new circle, we sample candidate positions in the free space.
        # A simple heuristic: sample points in the rectangle, check distance to all placed circles.
        # Max radius at (x, y) is min(
        #     x, W-x, y, H-y,
        #     min_{j in placed} (dist((x,y), (xj,yj)) - rj)
        # )
        
        # To speed up, we can use a grid of candidate points or random sampling.
        # Given N=21, O(N^2) checks per candidate is fine if candidates are limited.
        
        # We will use a set of candidate points:
        # 1. Corners (0,0), (W,0), (0,H), (W,H) - offset by r
        # 2. Midpoints of edges
        # 3. Random points in the rectangle
        
        # However, a smarter approach is to sample points uniformly and keep the best.
        # Let's use a fixed number of random samples per circle to balance speed and quality.
        # 200 samples per circle * 21 circles = 4200 checks. Each check is O(N). Total ~ 4200*21 ~ 100k ops. Very fast.
        
        num_samples = 500
        
        for idx in order:
            current_time = time.time()
            if current_time > time_limit:
                return 0.0, None
            
            best_r = 0.0
            best_x = 0.0
            best_y = 0.0
            
            # Generate candidate points
            # Mix of random and structured points
            # 1. Random points
            xs = np.random.uniform(0, W, num_samples)
            ys = np.random.uniform(0, H, num_samples)
            
            # 2. Add some structured points (corners, center, etc.)
            # Corners (slightly inset)
            eps = 1e-6
            xs = np.concatenate([xs, [eps, W-eps, eps, W-eps, W/2, W/2]])
            ys = np.concatenate([ys, [eps, eps, H-eps, H-eps, H/2, H/2]])
            
            # Evaluate max radius for each candidate
            # r_max(x, y) = min(
            #   x, W-x, y, H-y,
            #   min_{j} (sqrt((x-xj)^2 + (y-yj)^2) - rj)
            # )
            
            # We can vectorize the distance calculation to placed circles.
            # Placed circles: list of (xj, yj, rj)
            if not placed:
                # First circle: max radius is min(W, H) / 2. Place at center.
                r = min(W, H) / 2.0
                x = W / 2.0
                y = H / 2.0
                placed.append((x, y, r))
                continue
            
            placed_arr = np.array(placed) # shape (M, 3)
            px = placed_arr[:, 0]
            py = placed_arr[:, 1]
            pr = placed_arr[:, 2]
            
            # Distances from candidate points (xs, ys) to placed circles
            # dist[i, j] = sqrt((xs[i] - px[j])^2 + (ys[i] - py[j])^2)
            # We want min_j (dist[i, j] - pr[j])
            
            # Vectorized distance calculation
            # xs shape (S,), px shape (M,)
            # diff_x shape (S, M)
            diff_x = xs[:, None] - px[None, :]
            diff_y = ys[:, None] - py[None, :]
            dists = np.sqrt(diff_x**2 + diff_y**2)
            
            # Subtract radii
            dists_minus_r = dists - pr[None, :]
            
            # Min over placed circles (axis 1)
            min_dist_minus_r = np.min(dists_minus_r, axis=1)
            
            # Wall constraints
            wall_r = np.minimum(np.minimum(xs, W - xs), np.minimum(ys, H - ys))
            
            # Max radius is min(wall_r, min_dist_minus_r)
            max_r = np.minimum(wall_r, min_dist_minus_r)
            
            # Find the candidate with the largest max_r
            # Note: max_r can be negative if the point is inside a circle. We want max(0, max_r) effectively, 
            # but if all are negative, we fail to place? No, we just pick the best (least negative/positive).
            # If best_r <= 0, we might still place a tiny circle or fail.
            # Let's pick the index of the max.
            best_idx = np.argmax(max_r)
            best_r = max_r[best_idx]
            
            if best_r < 1e-4:
                # Couldn't place a meaningful circle. 
                # This can happen if the space is too crowded.
                # We can try to place a very small one or fail.
                # For now, if it's too small, we might want to fail this restart to try a different order/aspect.
                # But let's allow very small circles to complete the set, as long as they fit.
                pass
            
            best_x = xs[best_idx]
            best_y = ys[best_idx]
            
            # Ensure strict validity: clip r to be slightly less than calculated to avoid float errors
            # The calculated best_r is the theoretical max. 
            # We use best_r directly.
            
            placed.append((best_x, best_y, best_r))
            
        # After placing all N circles, check validity
        # The greedy placement ensures no overlap by construction (using max_r).
        # But float precision might cause issues.
        # Let's do a quick check.
        config = np.array(placed)
        
        # Verify perimeter
        # W and H are fixed for this restart.
        # Perimeter = 2*(W+H). We need 2*(W+H) <= 4 => W+H <= 2.
        # In our search, we will fix W+H=2.
        
        # Verify bounds and overlaps with tolerance
        for i in range(N):
            x, y, r = config[i]
            if x < r - 1e-5 or x > W - r + 1e-5 or y < r - 1e-5 or y > H - r + 1e-5:
                return 0.0, None
        
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, r1 = config[i]
                x2, y2, r2 = config[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                min_dist = r1 + r2
                if dist_sq < min_dist*min_dist - 1e-5:
                    return 0.0, None
        
        sum_r = np.sum(config[:, 2])
        return sum_r, config

    # Main Search Loop
    # We vary the aspect ratio W/H.
    # Let W + H = 2.
    # W ranges from 0.5 to 1.5 (symmetric, so 0.5 to 1.0 is enough, but let's go 0.4 to 1.6 to be safe? No, W,H > 0).
    # Actually, due to symmetry, we can just iterate W from 0.5 to 1.0.
    
    np.random.seed(42)
    random.seed(42)
    
    # Define a set of aspect ratios to try
    # Uniform distribution of W in [0.5, 1.0]
    # Let's try 20 different aspect ratios.
    W_values = np.linspace(0.5, 1.0, 20)
    
    # For each W, try several random insertion orders.
    orders_per_W = 10
    
    for W in W_values:
        H = 2.0 - W
        if H <= 0:
            continue
        
        for restart in range(orders_per_W):
            current_time = time.time()
            if current_time > deadline - 5.0:
                break
            
            # Random permutation of circle indices
            order = list(range(N))
            random.shuffle(order)
            
            # Time limit for this specific packing attempt
            # We have ~140s total. 20 W * 10 restarts = 200 attempts.
            # 140 / 200 = 0.7s per attempt.
            # Let's give 0.5s per attempt.
            t_limit = current_time + 0.5
            
            sum_r, config = try_packing(W, H, order, t_limit)
            
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_config = config
                
            # Early termination if we find a very good solution?
            # Theoretical max for 21 circles in 2x1 rect?
            # If we find > 3.0, it's likely optimal or close.
            # Let's not hardcode, but if we have time, we continue.

    if best_config is not None:
        return best_config
    
    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    W = 1.0
    H = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) * W / side
            y = (j + 0.5) * H / side
            r = 0.5 * min(W, H) / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
