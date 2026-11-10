import numpy as np
import time
import random

N = 30
# Precompute pairs for each i? 
# Or just iterate j, k in the loop.
# Indices 0..29
# Pairs (j, k) with j < k.
# Total pairs 435.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # 1. Generate Initial Candidates
    # Grids are usually best.
    def make_grid(w, h, ox, oy):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x + ox, y + oy))
        return pts
        
    inits = []
    # 5x6
    inits.append(make_grid(5, 6, -2, -2))
    # 6x5
    inits.append(make_grid(6, 5, -2, -2))
    # 3x10
    inits.append(make_grid(3, 10, -1, -4))
    # 10x3
    inits.append(make_grid(10, 3, -4, -1))
    # 2x15
    inits.append(make_grid(2, 15, -0, -7)) # Centering 2x15: x in {0,1}, y in {0..14} -> x-0.5? No integer. 
    # Let's just use 5x6 and 6x5 as primary.
    
    best_score = 0
    best_points = None
    
    # Helper to initialize counts and max
    def init_counts(points):
        counts = np.zeros(2000, dtype=np.int32) # Max area is bounded. 
        # Max diff in x, y is 60. 
        # Det = x1*y2 - x2*y1. Max |det| roughly 60*60 + 60*60 = 7200? 
        # Actually points are in [-30, 30]. 
        # Vector v1 = p2 - p1, v2 = p3 - p1.
        # Components of v1, v2 are in [-60, 60].
        # Det = v1x*v2y - v1y*v2x.
        # Max |det| = 60*60 + 60*60 = 7200.
        # So array size 7201 is safe.
        counts = np.zeros(7201, dtype=np.int32)
        
        for i in range(N):
            x1, y1 = points[i]
            for j in range(i + 1, N):
                x2, y2 = points[j]
                for k in range(j + 1, N):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        counts[det] += 1
        max_val = counts.max()
        return counts, int(max_val)
        
    # Evaluate inits
    for init in inits:
        if len(set(init)) != N:
            continue
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in init):
            continue
        counts, score = init_counts(init)
        if score > best_score:
            best_score = score
            best_points = init[:]
            
    if best_points is None:
        # Fallback random
        best_points = []
        while len(best_points) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in best_points:
                best_points.append(p)
        counts, best_score = init_counts(best_points)

    # 2. Local Search with Incremental Updates
    points = best_points[:]
    counts, current_max = init_counts(points)
    best_found_score = best_score
    best_found_points = best_points[:]
    
    # Precompute list of all pairs (j, k)
    all_pairs = []
    for j in range(N):
        for k in range(j + 1, N):
            all_pairs.append((j, k))
            
    # We need to filter pairs based on i. 
    # Instead of filtering every time, we can just check j != i and k != i inside the loop.
    
    while time.time() < deadline:
        # Pick a point to move
        idx = random.randint(0, N - 1)
        old_point = points[idx]
        
        # Generate new point
        # Mix of local and global moves
        if random.random() < 0.7:
            # Local move
            new_x = old_point[0] + random.randint(-3, 3)
            new_y = old_point[1] + random.randint(-3, 3)
        else:
            # Global move
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        new_point = (new_x, new_y)
        
        # Check bounds
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        # Check distinctness
        if new_point == old_point:
            continue
        is_valid = True
        for i in range(N):
            if i != idx and points[i] == new_point:
                is_valid = False
                break
        if not is_valid:
            continue
            
        # Apply Move: Update counts
        # We need to remove contributions of old_point and add new_point.
        # For all pairs (j, k) where j != idx and k != idx
        
        # Optimization: We can just iterate all pairs and skip if j==idx or k==idx
        # This is 435 iterations.
        
        # Save old counts? No, we revert if needed.
        # We will modify `counts` in place.
        
        # To speed up, we can pre-compute the pairs that do NOT involve idx?
        # That requires building a list every time or having a pre-computed structure.
        # Let's just loop.
        
        # Actually, we can vectorize the area calculation for the specific idx?
        # No, let's stick to Python loop for simplicity and correctness, it's fast enough.
        
        # Store the changes to allow fast revert?
        # Or just re-apply the reverse move.
        
        # Let's do a "dry run" style: 
        # 1. Calculate new counts for affected areas? 
        # No, just update `counts` directly.
        
        # To make revert easy, we can store the list of (area, delta) changes.
        # But that's overhead.
        # Let's just update and if rejected, update back.
        
        # Update
        ox, oy = old_point
        nx, ny = new_point
        
        # We need to be careful with the max tracking.
        # We will just compute max(counts) at the end of the move.
        
        for j, k in all_pairs:
            if j == idx or k == idx:
                continue
            
            xj, yj = points[j]
            xk, yk = points[k]
            
            # Area with old point
            det_old = abs((xj - ox) * (yk - oy) - (yj - oy) * (xk - ox))
            if det_old > 0:
                counts[det_old] -= 1
                
            # Area with new point
            det_new = abs((xj - nx) * (yk - ny) - (yj - ny) * (xk - nx))
            if det_new > 0:
                counts[det_new] += 1
            elif det_old > 0:
                # If det_new is 0, we just decremented det_old. 
                # Nothing to add.
                pass
            # If det_old was 0 and det_new > 0, we added det_new.
            # If det_old > 0 and det_new == 0, we removed det_old.
            
        new_max = int(counts.max())
        
        if new_max > current_max:
            # Accept
            points[idx] = new_point
            current_max = new_max
            if current_max > best_found_score:
                best_found_score = current_max
                best_found_points = points[:]
        elif new_max == current_max:
            # Accept sideways with some probability
            if random.random() < 0.5:
                points[idx] = new_point
            else:
                # Revert
                for j, k in all_pairs:
                    if j == idx or k == idx:
                        continue
                    xj, yj = points[j]
                    xk, yk = points[k]
                    det_old = abs((xj - nx) * (yk - ny) - (yj - ny) * (xk - nx)) # Current is new
                    if det_old > 0:
                        counts[det_old] -= 1
                    det_new = abs((xj - ox) * (yk - oy) - (yj - oy) * (xk - ox)) # Target is old
                    if det_new > 0:
                        counts[det_new] += 1
        else:
            # Revert
            for j, k in all_pairs:
                if j == idx or k == idx:
                    continue
                xj, yj = points[j]
                xk, yk = points[k]
                det_old = abs((xj - nx) * (yk - ny) - (yj - ny) * (xk - nx))
                if det_old > 0:
                    counts[det_old] -= 1
                det_new = abs((xj - ox) * (yk - oy) - (yj - oy) * (xk - ox))
                if det_new > 0:
                    counts[det_new] += 1

    return [(int(p[0]), int(p[1])) for p in best_found_points]
