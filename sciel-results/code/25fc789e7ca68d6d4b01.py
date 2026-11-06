import time
import itertools
import random

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin
    
    # Helper to compute doubled area (absolute determinant)
    def doubled_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    
    # Helper to compute histogram and max multiplicity
    def compute_score(points):
        counts = {}
        for i, j, k in itertools.combinations(range(len(points)), 3):
            a = doubled_area(points[i], points[j], points[k])
            if a > 0:
                counts[a] = counts.get(a, 0) + 1
        if not counts:
            return 0, counts
        return max(counts.values()), counts

    # Generate initial candidates
    candidates = []
    
    # 1. Rectangular Grids
    # 5x6
    pts = [(x, y) for x in range(5) for y in range(6)]
    # Center them roughly to allow movement
    cx, cy = -2, -2
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)
    
    # 3x10
    pts = [(x, y) for x in range(3) for y in range(10)]
    cx, cy = -1, -4
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)
    
    # 2x15
    pts = [(x, y) for x in range(2) for y in range(15)]
    cx, cy = -0, -7
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)
    
    # 6x5
    pts = [(x, y) for x in range(6) for y in range(5)]
    cx, cy = -2, -2
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)
    
    # 2. Sheared Grids (Affine transformations with det 1)
    # Take 5x6 grid, apply shear
    base = [(x, y) for x in range(5) for y in range(6)]
    for k in [1, -1, 2, -2]:
        pts = [(x, y + k*x) for x, y in base]
        # Check bounds
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
             # Center
             xs = [p[0] for p in pts]
             ys = [p[1] for p in pts]
             cx = (min(xs) + max(xs)) // 2
             cy = (min(ys) + max(ys)) // 2
             pts = [(x - cx, y - cy) for x, y in pts]
             candidates.append(pts)

    best_points = None
    best_score = 0
    best_hist = None
    
    # Evaluate initial candidates
    for pts in candidates:
        # Ensure distinct
        if len(set(pts)) != 30: continue
        score, hist = compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            best_hist = hist
    
    if best_points is None:
        return candidates[0]
        
    # Local Search
    # We will try to improve the best score
    # State: points, hist, score
    
    # To speed up, we only update the histogram for triangles involving the moved point
    # Note: This function MODIFIES the passed hist. Caller must pass a copy if rollback is needed.
    def update_histogram(points, hist, moved_idx, old_point, new_point):
        # Remove old triangles involving moved_idx
        n = len(points)
        for j in range(n):
            if j == moved_idx: continue
            for k in range(j+1, n):
                if k == moved_idx: continue
                a = doubled_area(old_point, points[j], points[k])
                if a > 0:
                    if a in hist:
                        hist[a] -= 1
                        if hist[a] == 0:
                            del hist[a]
                    else:
                        # This should not happen if hist is consistent, but guard against it
                        pass
        
        # Add new triangles involving moved_idx
        for j in range(n):
            if j == moved_idx: continue
            for k in range(j+1, n):
                if k == moved_idx: continue
                a = doubled_area(new_point, points[j], points[k])
                if a > 0:
                    hist[a] = hist.get(a, 0) + 1
                    
        # Calculate new max
        if not hist:
             return 0
        return max(hist.values())

    # Define neighbor positions
    # We'll try moving a point to any valid integer coordinate in a small radius
    # Radius 2 or 3 is probably enough
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        points = best_points[:]
        hist = best_hist.copy()
        current_score = best_score
        
        for i in range(30):
            if (time.time() - start_time) > time_limit:
                break
            old_p = points[i]
            # Try neighbors
            # Generate neighbors in a 5x5 box around old_p
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0: continue
                    new_p = (old_p[0] + dx, old_p[1] + dy)
                    # Check bounds
                    if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
                        continue
                    # Check distinctness
                    if new_p in points and points.index(new_p) != i:
                        continue
                    # Actually, check if new_p is already in points
                    # Since we are replacing points[i], new_p must not be in points[j] for j != i
                    is_duplicate = False
                    for j in range(30):
                        if j != i and points[j] == new_p:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue
                        
                    # Update histogram
                    # We must use a copy of hist because update_histogram modifies it.
                    # If the move is rejected, we need the original hist.
                    hist_trial = hist.copy()
                    new_score = update_histogram(points, hist_trial, i, old_p, new_p)
                    
                    if new_score > current_score:
                        # Accept move
                        points[i] = new_p
                        hist = hist_trial
                        current_score = new_score
                        improved = True
                        # Break out of neighbor loop to restart with new state?
                        # Or just continue?
                        # Breaking and restarting the i-loop is safer to avoid stale hist
                        break
                else:
                    continue
                break
            if improved:
                # Restart the outer loop with the new points
                best_points = points
                best_hist = hist
                best_score = current_score
                break
                
    # Final check
    if (time.time() - start_time) < time_limit:
        # Maybe try a random restart?
        # Or just return best found
        pass
        
    return best_points
