import time
import random
import math
import numpy as np

N = 64
TIME_LIMIT = 280

# Precompute distance matrix
# Index mapping: idx = x * N + y
def get_idx(x, y):
    return x * N + y

def get_xy(idx):
    return (idx // N, idx % N)

# Build distance matrix
points = [(x, y) for x in range(N) for y in range(N)]
num_pts = N * N
xs = np.array([p[0] for p in points])
ys = np.array([p[1] for p in points])
# D[i, j] = dist2(points[i], points[j])
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
D = (dx * dx + dy * dy).astype(np.int32)

# Known Golomb ruler of order 10 (length 55)
# Marks: 0, 1, 4, 10, 18, 23, 30, 39, 47, 55
golomb_marks = [0, 1, 4, 10, 18, 23, 30, 39, 47, 55]
golomb_pts_x = [get_idx(m, 0) for m in golomb_marks]
golomb_pts_y = [get_idx(0, m) for m in golomb_marks]

def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    
    def can_add(p_idx, pts_indices):
        """Check if point p_idx can be added to pts_indices."""
        k = len(pts_indices)
        if k == 0:
            return True
        
        # Get distances from p_idx to all pts in pts_indices
        # D is (num_pts, num_pts)
        dists = D[p_idx, pts_indices]
        
        # Check 1: Distances from p to others must be distinct
        if len(set(dists)) != k:
            return False
        
        d_set = set(dists)
        
        # Check 2: No existing pair (i, j) has distance equal to any dist from p
        # i.e., d(i, j) not in d_set
        for i in range(k):
            idx_i = pts_indices[i]
            # Only check j > i to avoid duplicates
            for j in range(i + 1, k):
                idx_j = pts_indices[j]
                if D[idx_i, idx_j] in d_set:
                    return False
        
        return True

    def greedy_construct(initial_indices, rng):
        """Greedy construction starting from initial_indices."""
        pts_indices = list(initial_indices)
        
        # Generate candidate list of all other points
        all_indices = set(range(num_pts))
        for idx in pts_indices:
            all_indices.discard(idx)
        
        candidates = list(all_indices)
        rng.shuffle(candidates)
        
        for idx in candidates:
            if can_add(idx, pts_indices):
                pts_indices.append(idx)
            # Time check
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return pts_indices

    def local_improvement(pts_indices, rng, max_iters=100):
        """Try to add more points."""
        pts_indices = list(pts_indices)
        all_indices = set(range(num_pts))
        for idx in pts_indices:
            all_indices.discard(idx)
        
        candidates = list(all_indices)
        rng.shuffle(candidates)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for idx in candidates:
                if can_add(idx, pts_indices):
                    pts_indices.append(idx)
                    candidates.remove(idx)
                    improved = True
                    break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return pts_indices

    # 1. Try structured seeds (Golomb rulers)
    seeds = [
        golomb_pts_x,
        golomb_pts_y,
        # Maybe a mix?
        golomb_pts_x[:5] + golomb_pts_y[:5] # Not necessarily valid, but greedy will fix/keep valid ones
    ]
    
    for seed in seeds:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        rng = random.Random(42)
        pts = greedy_construct(seed, rng)
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts
        # Local improvement
        pts = local_improvement(pts, rng)
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts

    # 2. Random restarts
    # Estimate time per restart. 
    # Let's run as many as we can.
    restart_count = 0
    while time.time() - start_time < TIME_LIMIT - 10:
        rng = random.Random(10000 + restart_count * 12345)
        pts = greedy_construct([], rng)
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts
        
        # Local improvement
        pts = local_improvement(pts, rng)
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts
            
        restart_count += 1
        # Optional: if best_len >= 10, we might want to focus on finding 11+
        # But random restarts are cheap.

    # Convert best_pts indices back to (x, y) tuples
    result = [get_xy(idx) for idx in best_pts]
    return result
