import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - t0

    def compute_max_area_count_fast(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        Optimized using pairwise difference vectors and numpy.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]  # shape (M, 2), M = n*(n-1)/2
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)

        # For each base vector a (index u), compute cross products with all later vectors v
        # Use a faster approach: collect all cross products, then count
        all_cross = np.empty(0, dtype=np.int64)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) > 0:
                all_cross = np.concatenate([all_cross, np.abs(nz)])

        if len(all_cross) == 0:
            return 0
        
        vals, cnts = np.unique(all_cross, return_counts=True)
        return int(cnts.max())

    def score_of(pts):
        return compute_max_area_count_fast(pts)

    # 1. Initialize with strong structured candidates
    
    # Candidate 1: Two lines, 15 points each
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    
    # Candidate 2: Grid 6x5
    cand2 = [(x, y) for x in range(6) for y in range(5)]
    
    # Candidate 3: Three lines, 10 points each
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
            
    # Candidate 5: Four lines with varying spacing
    cand5 = [(x, 0) for x in range(8)] + [(x, 2) for x in range(8)] + \
            [(x, 4) for x in range(7)] + [(x, 6) for x in range(7)]
    
    # Candidate 6: Concentric-like structure
    cand6 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + \
            [(x, -1) for x in range(5)] + [(x, -2) for x in range(5)]
    
    # Candidate 7: 5 lines, 6 points each
    cand7 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
            [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
            [(x, 4) for x in range(6)]
    
    # Candidate 8: 2 lines with offset
    cand8 = [(x, 0) for x in range(15)] + [(x+1, 1) for x in range(14)] + [(14, 1)]
    # Ensure distinctness and bounds for cand8
    cand8 = []
    used8 = set()
    for x in range(15):
        pt = (x, 0)
        if pt not in used8:
            cand8.append(pt)
            used8.add(pt)
    for x in range(15):
        pt = (x, 1)
        if pt not in used8:
            cand8.append(pt)
            used8.add(pt)
    # If less than 30, fill randomly
    if len(cand8) < 30:
        while len(cand8) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used8:
                cand8.append((x, y))
                used8.add((x, y))

    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8]
    
    best_points = None
    best_score = 0
    
    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    # 2. Local Search with improved strategy
    current_points = list(best_points)
    current_score = best_score
    
    random.seed(42)
    step_size = 2
    
    # Track if we've made progress recently
    last_improvement_iter = 0
    iter_count = 0
    no_improve_count = 0
    used_set = set(current_points)
    
    while elapsed() < DEADLINE:
        iter_count += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        # Mix of small steps and occasional larger jumps
        r = random.random()
        if r < 0.01:
            # Large jump - full random reposition
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.08:
            # Medium jump
            new_x = old_x + random.randint(-4, 4)
            new_y = old_y + random.randint(-4, 4)
        else:
            # Small step
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)
            
        # Ensure bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check distinctness
        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue
        if new_pt in used_set:
            continue
            
        # Try the move
        used_set.discard((old_x, old_y))
        used_set.add(new_pt)
        current_points[idx] = new_pt
        
        new_score = score_of(current_points)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                last_improvement_iter = iter_count
                # Decay step size on improvement to refine
                if step_size > 1 and random.random() < 0.2:
                    step_size = max(1, step_size - 1)
        else:
            # Revert
            used_set.discard(new_pt)
            used_set.add((old_x, old_y))
            current_points[idx] = (old_x, old_y)
            no_improve_count += 1
            
        # If stuck for a long time, restart from best
        if no_improve_count > 1500:
            # Restart from best points
            current_points = list(best_points)
            current_score = best_score
            no_improve_count = 0
            last_improvement_iter = iter_count
            used_set = set(current_points)
            # Increase step size to explore more
            step_size = random.randint(3, 8)
            
        # Occasionally adjust step size
        if random.random() < 0.005:
            step_size = random.randint(1, 4)

    # Final polish: try to improve by moving individual points to nearby positions
    if elapsed() < DEADLINE - 5:
        polish_points = list(best_points)
        polish_score = best_score
        polish_used = set(polish_points)
        
        for _ in range(200):
            if elapsed() > DEADLINE - 5:
                break
            idx = random.randint(0, 29)
            old_x, old_y = polish_points[idx]
            
            # Try all nearby positions
            best_new = None
            best_new_score = polish_score
            
            for dx_p in range(-2, 3):
                for dy_p in range(-2, 3):
                    if dx_p == 0 and dy_p == 0:
                        continue
                    nx, ny = old_x + dx_p, old_y + dy_p
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    npt = (nx, ny)
                    if npt in polish_used:
                        continue
                    
                    # Try this position
                    polish_used.discard((old_x, old_y))
                    polish_used.add(npt)
                    polish_points[idx] = npt
                    s = score_of(polish_points)
                    
                    if s > best_new_score:
                        best_new_score = s
                        best_new = npt
                    else:
                        # Revert for next try
                        polish_used.discard(npt)
                        polish_used.add((old_x, old_y))
                        polish_points[idx] = (old_x, old_y)
            
            if best_new is not None:
                polish_used.discard((old_x, old_y))
                polish_used.add(best_new)
                polish_points[idx] = best_new
                polish_score = best_new_score
                if polish_score > best_score:
                    best_score = polish_score
                    best_points = list(polish_points)
            else:
                # Ensure we're back to original state
                polish_used.discard((old_x, old_y))
                polish_used.add((old_x, old_y))
                polish_points[idx] = (old_x, old_y)
    
    return best_points
