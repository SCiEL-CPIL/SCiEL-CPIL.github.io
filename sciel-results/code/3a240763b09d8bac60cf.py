import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - t0

    def compute_max_area_count(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        Uses a single vectorized pass over all O(n^2) point pairs.
        pts: list of 30 (x, y) tuples.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]  # shape (M, 2), M = n*(n-1)/2
        dx = d[:, 0].astype(np.int64)
        dy = d[:, 1].astype(np.int64)
        M = len(dx)

        # For each base vector a (index u), compute cross products with all later vectors v
        # Cross(u, v) = dx[u]*dy[v] - dy[u]*dx[v]
        # For fixed u, the vector of cross products over v=u+1..M-1 is:
        #   dx[u]*dy[u+1:] - dy[u]*dx[u+1:]
        # Accumulate counts per area value.
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            # Count absolute values
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)

        if not area_counts:
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count(pts)

    # 1. Initialize with strong structured candidates
    
    # Candidate 1: Two lines, 15 points each
    # y=0, x=0..14
    # y=1, x=0..14
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    
    # Candidate 2: Grid 6x5
    cand2 = [(x, y) for x in range(6) for y in range(5)]
    
    # Candidate 3: Three lines, 10 points each
    # y=0, x=0..9
    # y=1, x=0..9
    # y=2, x=0..9
    cand3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    
    # Candidate 4: Random
    random.seed(42)
    cand4 = []
    used = set()
    while len(cand4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            cand4.append((x, y))
            
    candidates = [cand1, cand2, cand3, cand4]
    
    best_points = None
    best_score = 0
    
    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    # 2. Local Search
    current_points = list(best_points)
    current_score = best_score
    
    random.seed(42)
    step_size = 3
    
    # Track if we've made progress recently
    last_improvement_iter = 0
    iter_count = 0
    
    while elapsed() < DEADLINE:
        iter_count += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        # Mix of small steps and occasional larger jumps
        r = random.random()
        if r < 0.03:
            # Large jump - full random reposition
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.15:
            # Medium jump
            new_x = old_x + random.randint(-6, 6)
            new_y = old_y + random.randint(-6, 6)
        else:
            # Small step
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)
            
        # Ensure bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check distinctness using set for speed
        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue
        is_valid = True
        for j in range(30):
            if j == idx:
                continue
            if current_points[j] == new_pt:
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Try the move
        current_points[idx] = new_pt
        
        new_score = score_of(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                last_improvement_iter = iter_count
                # Decay step size on improvement to refine
                if step_size > 1 and random.random() < 0.4:
                    step_size = max(1, step_size - 1)
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            
        # If stuck for a long time, restart from best or random
        if iter_count - last_improvement_iter > 3000:
            # Restart from best points
            current_points = list(best_points)
            current_score = best_score
            last_improvement_iter = iter_count
            # Increase step size to explore more
            step_size = random.randint(4, 12)
            
        # Occasionally adjust step size
        if random.random() < 0.005:
            step_size = random.randint(1, 6)

    return best_points
