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
        Vectorized approach: for each pair (i,j), the doubled area of triangle (i,j,k)
        is |cross(P[j]-P[i], P[k]-P[i])| = |cross(P[j]-P[i], P[k]-P[j]) + cross(P[j]-P[i], P[j]-P[i])|
        Actually simpler: doubled area of (i,j,k) = |det([P[j]-P[i], P[k]-P[i]])|
        We can compute this as: for each ordered pair (i,j) with i<j, and each k not in {i,j},
        area = |cross(P[j]-P[i], P[k]-P[i])|
        
        A faster approach: 
        - Precompute all pairwise differences
        - For each vector v = P[j]-P[i], compute cross with all other vectors w = P[k]-P[i]
        - But this is still O(n^3)
        
        Alternative: Use the fact that doubled area of triangle (i,j,k) = 
        |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        
        Let's use a batched approach with numpy broadcasting for chunks.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)
        n = len(P)
        
        # Compute all triple determinants efficiently
        # For each i, compute cross products of (P[j]-P[i], P[k]-P[i]) for all j,k > i
        # Group by i to keep memory reasonable
        
        max_count = 0
        areas_list = []
        
        for i in range(n - 2):
            # Vectors from P[i] to all other points
            diffs = P[i+1:] - P[i]  # shape (n-1-i, 2)
            m = len(diffs)
            if m < 2:
                continue
            # Cross product of each pair of vectors
            # cross(diffs[a], diffs[b]) for a < b
            # Use broadcasting: diffs[:, 0:1] * diffs[1:, 1] - diffs[:, 1:2] * diffs[1:, 0]
            # This gives shape (m-1, m-1), we want upper triangle
            dx = diffs[:, 0]
            dy = diffs[:, 1]
            # For pairs (a,b) with a < b:
            # cross = dx[a]*dy[b] - dy[a]*dx[b]
            # Vectorized: outer product
            cross_mat = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
            # Take upper triangle (a < b means row < col in the (m-1)x(m-1) matrix if we index properly)
            # Actually diffs has m elements, pairs (a,b) with 0<=a<b<m
            # cross_mat[a,b] = |dx[a]*dy[b] - dy[a]*dx[b]| for a != b
            # We want a < b, so take upper triangle
            tri = np.triu(cross_mat, k=1)
            nz = tri[tri != 0]
            if len(nz) > 0:
                areas_list.append(nz)
        
        if not areas_list:
            return 0
        
        all_areas = np.concatenate(areas_list)
        if len(all_areas) == 0:
            return 0
        
        vals, cnts = np.unique(all_areas, return_counts=True)
        return int(cnts.max())

    def score_of(pts):
        return compute_max_area_count_fast(pts)

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
            
    # 2. Local Search with improved strategy
    current_points = list(best_points)
    current_score = best_score
    
    random.seed(42)
    step_size = 2
    
    # Track progress
    iter_count = 0
    no_improve_count = 0
    used_set = set(current_points)
    
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
        if no_improve_count > 800:
            # Restart from best points
            current_points = list(best_points)
            current_score = best_score
            no_improve_count = 0
            used_set = set(current_points)
            # Increase step size to explore more
            step_size = random.randint(4, 10)
            
        # Occasionally adjust step size
        if random.random() < 0.003:
            step_size = random.randint(1, 5)
    
    # Final polish: try to improve by moving individual points to nearby positions
    if elapsed() < DEADLINE - 1:
        polish_points = list(best_points)
        polish_score = best_score
        polish_used = set(polish_points)
        
        polish_iters = 0
        while elapsed() < DEADLINE - 0.5 and polish_iters < 500:
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
