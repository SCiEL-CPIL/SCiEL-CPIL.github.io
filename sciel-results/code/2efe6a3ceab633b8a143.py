import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Helper to evaluate a set of points
    def get_max_area_freq(points):
        areas = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    # Doubled area
                    d = abs(x1*(y2-y3) - y1*(x2-x3) + x3*(y2-y1))
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())

    # Helper to generate grid points
    def make_grid(rows, cols, offset_x=0, offset_y=0):
        pts = []
        for r in range(rows):
            for c in range(cols):
                pts.append((offset_x + c, offset_y + r))
        return pts

    # Initial candidates
    candidates = []
    # 6x5
    candidates.append(make_grid(5, 6, -2, -2))
    # 5x6
    candidates.append(make_grid(6, 5, -2, -2))
    # 10x3
    candidates.append(make_grid(3, 10, -4, -1))
    # 15x2
    candidates.append(make_grid(2, 15, -7, 0))
    # 4x8 (take first 30)
    grid_4x8 = make_grid(8, 4, -3, -1)
    candidates.append(grid_4x8[:30])
    
    best_score = 0
    best_points = None
    
    for init_pts in candidates:
        # Ensure distinct and in bounds
        init_pts = list(set(init_pts))
        if len(init_pts) < 30:
            continue
            
        current_score = get_max_area_freq(init_pts)
        if current_score > best_score:
            best_score = current_score
            best_points = init_pts[:]
            
        # Local Search
        # We will use a simple hill climbing with random restarts within the time budget
        # To allow multiple restarts, we limit the time per grid or just break if time is low.
        
        # Incremental evaluation logic
        # We need to maintain the Counter and the points list.
        
        def local_search(points):
            nonlocal best_score, best_points
            
            # Precompute all triangle areas and indices?
            # Or just use the incremental update.
            
            # Let's use a slightly more robust local search:
            # 1. Calculate current histogram.
            # 2. Loop:
            #    - Pick random i
            #    - Pick random neighbor (dx, dy) in [-2, 2]
            #    - Check validity
            #    - Update histogram
            #    - Check max
            #    - Accept/Reject
            
            hist = Counter()
            n = len(points)
            # Fill hist
            for i in range(n):
                x1, y1 = points[i]
                for j in range(i+1, n):
                    x2, y2 = points[j]
                    for k in range(j+1, n):
                        x3, y3 = points[k]
                        d = abs(x1*(y2-y3) - y1*(x2-x3) + x3*(y2-y1))
                        if d > 0:
                            hist[d] += 1
            
            current_max = max(hist.values()) if hist else 0
            
            # If this initial grid is already better, update global best
            if current_max > best_score:
                best_score = current_max
                best_points = points[:]
                
            stagnation = 0
            max_stagnation = 100
            
            while time.time() < deadline - 2.0:
                if stagnation > max_stagnation:
                    break
                    
                i = random.randint(0, n-1)
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                
                old_x, old_y = points[i]
                new_x, new_y = old_x + dx, old_y + dy
                
                # Bounds check [-30, 30]
                if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                    continue
                
                # Collision check
                is_collision = False
                for j in range(n):
                    if j == i: continue
                    if points[j] == (new_x, new_y):
                        is_collision = True
                        break
                if is_collision:
                    continue
                
                # Update histogram
                # Remove old triangles involving i
                to_remove = []
                to_add = []
                
                x1_old, y1_old = old_x, old_y
                x1_new, y1_new = new_x, new_y
                
                for j in range(n):
                    if j == i: continue
                    x2, y2 = points[j]
                    for k in range(j+1, n):
                        if k == i: continue
                        # We need to be careful with indices.
                        # The triangle is (i, j, k) or (j, k, i) etc.
                        # It's easier to just iterate all pairs (j, k) with j != i, k != i, j < k
                        pass
                
                # Simpler approach for incremental update:
                # Iterate all j, k such that {i, j, k} is a set.
                # There are (n-1)*(n-2)/2 such pairs.
                
                old_areas = []
                new_areas = []
                
                for j in range(n):
                    if j == i: continue
                    x2, y2 = points[j]
                    for k in range(j+1, n):
                        if k == i: continue
                        x3, y3 = points[k]
                        
                        # Old area
                        d_old = abs(x1_old*(y2-y3) - y1_old*(x2-x3) + x3*(y2-y1_old))
                        if d_old > 0:
                            old_areas.append(d_old)
                            
                        # New area
                        d_new = abs(x1_new*(y2-y3) - y1_new*(x2-x3) + x3*(y2-y1_new))
                        if d_new > 0:
                            new_areas.append(d_new)
                        
                        # Wait, the loop condition `k in range(j+1, n)` combined with `if k == i: continue`
                        # and `if j == i: continue` handles the pairs correctly?
                        # Yes, it iterates all unique pairs (j,k) from the remaining n-1 points.
                        
                # Apply changes
                for d in old_areas:
                    hist[d] -= 1
                    if hist[d] == 0:
                        del hist[d]
                        
                for d in new_areas:
                    hist[d] += 1
                    
                # Find new max
                new_max = max(hist.values()) if hist else 0
                
                if new_max > current_max:
                    points[i] = (new_x, new_y)
                    current_max = new_max
                    stagnation = 0
                    
                    if current_max > best_score:
                        best_score = current_max
                        best_points = points[:]
                else:
                    # Revert
                    # Since we modified hist, we must revert hist
                    for d in new_areas:
                        hist[d] -= 1
                        if hist[d] == 0:
                            del hist[d]
                    for d in old_areas:
                        hist[d] += 1
                    stagnation += 1

        local_search(init_pts[:])

    return best_points if best_points else candidates[0]
