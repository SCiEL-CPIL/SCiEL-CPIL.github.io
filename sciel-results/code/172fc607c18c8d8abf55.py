import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 57.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - t0

    def compute_max_area_count_fast(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        Uses the formula: doubled area of (i,j,k) = |x_i*(y_j-y_k) + x_j*(y_k-y_i) + x_k*(y_i-y_j)|
        Vectorized over all C(n,3) triples at once.
        """
        P = np.array(pts, dtype=np.int64)
        n = len(P)
        if n < 3:
            return 0
        
        # Generate all index triples (i, j, k) with i < j < k
        # Use itertools for memory efficiency, but vectorize the computation
        import itertools
        idx = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
        if len(idx) == 0:
            return 0
        
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        
        # doubled area = |x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)|
        xi, yi = P[i, 0], P[i, 1]
        xj, yj = P[j, 0], P[j, 1]
        xk, yk = P[k, 0], P[k, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        areas = areas[areas != 0]
        
        if len(areas) == 0:
            return 0
        
        vals, cnts = np.unique(areas, return_counts=True)
        return int(cnts.max())

    def score_of(pts):
        return compute_max_area_count_fast(pts)

    def get_top_areas(pts, top_k=5):
        """Get the top-k most frequent non-zero doubled areas and their counts."""
        P = np.array(pts, dtype=np.int64)
        n = len(P)
        if n < 3:
            return []
        
        import itertools
        idx = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
        if len(idx) == 0:
            return []
        
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        xi, yi = P[i, 0], P[i, 1]
        xj, yj = P[j, 0], P[j, 1]
        xk, yk = P[k, 0], P[k, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        areas = areas[areas != 0]
        
        if len(areas) == 0:
            return []
        
        vals, cnts = np.unique(areas, return_counts=True)
        order = np.argsort(cnts)[::-1]
        return [(int(vals[order[t]]), int(cnts[order[t]])) for t in range(min(top_k, len(order)))]

    # 1. Initialize with strong structured candidates
    
    candidates = []
    
    # Candidate 1: Two lines, 15 points each
    candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])
    
    # Candidate 2: Grid 6x5
    candidates.append([(x, y) for x in range(6) for y in range(5)])
    
    # Candidate 3: Three lines, 10 points each
    candidates.append([(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)])
    
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
    candidates.append(cand4)
    
    # Candidate 5: Four lines with varying spacing
    candidates.append([(x, 0) for x in range(8)] + [(x, 2) for x in range(8)] + 
                      [(x, 4) for x in range(7)] + [(x, 6) for x in range(7)])
    
    # Candidate 6: Concentric-like structure
    candidates.append([(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + 
                      [(x, -1) for x in range(5)] + [(x, -2) for x in range(5)])
    
    # Candidate 7: 5 lines, 6 points each
    candidates.append([(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + 
                      [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + 
                      [(x, 4) for x in range(6)])
    
    # Candidate 8: 2 lines with offset
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
    if len(cand8) < 30:
        while len(cand8) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used8:
                cand8.append((x, y))
                used8.add((x, y))
    candidates.append(cand8)
    
    # Candidate 9: 3x10 grid
    candidates.append([(x, y) for x in range(10) for y in range(3)])
    
    # Candidate 10: 10x3 grid
    candidates.append([(x, y) for x in range(3) for y in range(10)])
    
    # Candidate 11: Arithmetic progression on two lines with different slopes
    cand11 = [(x, 0) for x in range(15)] + [(2*x, 1) for x in range(15)]
    # Check bounds and distinctness
    valid11 = []
    seen11 = set()
    for pt in cand11:
        if -30 <= pt[0] <= 30 and -30 <= pt[1] <= 30 and pt not in seen11:
            valid11.append(pt)
            seen11.add(pt)
    if len(valid11) >= 30:
        candidates.append(valid11[:30])
    
    # Candidate 12: Points on parabola y=x^2/4 scaled
    cand12 = []
    seen12 = set()
    for x in range(-12, 13):
        y = x * x // 4
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen12:
            cand12.append((x, y))
            seen12.add((x, y))
    if len(cand12) >= 30:
        candidates.append(cand12[:30])
    
    # Candidate 13: 4 lines, 7-8 points each
    candidates.append([(x, 0) for x in range(8)] + [(x, 2) for x in range(8)] + 
                      [(x, 4) for x in range(7)] + [(x, 6) for x in range(7)])
    
    # Candidate 14: 6 lines, 5 points each
    candidates.append([(x, 0) for x in range(5)] + [(x, 1) for x in range(5)] + 
                      [(x, 2) for x in range(5)] + [(x, 3) for x in range(5)] + 
                      [(x, 4) for x in range(5)] + [(x, 5) for x in range(5)])
    
    # Candidate 15: Diagonal lines
    cand15 = []
    seen15 = set()
    for d in range(-14, 15):
        for x in range(max(0, -d), min(30, 30-d) + 1):
            y = x + d
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen15:
                cand15.append((x, y))
                seen15.add((x, y))
                if len(cand15) >= 30:
                    break
        if len(cand15) >= 30:
            break
    if len(cand15) >= 30:
        candidates.append(cand15[:30])
    
    best_points = None
    best_score = 0
    
    for cand in candidates:
        if len(cand) != 30:
            continue
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    # 2. Local Search with target-area-guided strategy
    current_points = list(best_points)
    current_score = best_score
    
    random.seed(42)
    step_size = 2
    
    # Track progress
    iter_count = 0
    no_improve_count = 0
    used_set = set(current_points)
    
    # Get initial target areas
    top_areas = get_top_areas(current_points, top_k=3)
    target_area = top_areas[0][0] if top_areas else 1
    
    while elapsed() < DEADLINE - 3:
        iter_count += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate a new position with adaptive strategy
        r = random.random()
        if r < 0.005:
            # Very rare large jump
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.05:
            # Medium jump
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
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
                # Update target areas periodically
                if iter_count % 50 == 0:
                    top_areas = get_top_areas(current_points, top_k=3)
                    if top_areas:
                        target_area = top_areas[0][0]
                # Decay step size on improvement to refine
                if step_size > 1 and random.random() < 0.3:
                    step_size = max(1, step_size - 1)
        else:
            # Revert
            used_set.discard(new_pt)
            used_set.add((old_x, old_y))
            current_points[idx] = (old_x, old_y)
            no_improve_count += 1
            
        # If stuck, restart from best with larger exploration
        if no_improve_count > 600:
            # Restart from best points
            current_points = list(best_points)
            current_score = best_score
            no_improve_count = 0
            used_set = set(current_points)
            # Increase step size to explore more
            step_size = random.randint(4, 10)
            # Refresh target areas
            top_areas = get_top_areas(current_points, top_k=3)
            if top_areas:
                target_area = top_areas[0][0]
            
        # Occasionally adjust step size
        if random.random() < 0.003:
            step_size = random.randint(1, 5)
    
    # Final polish: try to improve by moving individual points to nearby positions
    if elapsed() < DEADLINE - 1:
        polish_points = list(best_points)
        polish_score = best_score
        polish_used = set(polish_points)
        
        polish_iters = 0
        while elapsed() < DEADLINE - 0.5 and polish_iters < 800:
            polish_iters += 1
            idx = random.randint(0, 29)
            old_x, old_y = polish_points[idx]
            
            # Try all nearby positions
            best_new = None
            best_new_score = polish_score
            
            for dx_p in range(-3, 4):
                for dy_p in range(-3, 4):
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
