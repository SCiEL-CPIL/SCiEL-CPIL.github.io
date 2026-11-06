import time
import itertools
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    # Helper to compute doubled area (absolute determinant)
    def doubled_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

    # Vectorized score computation
    def compute_score_vectorized(points):
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        # Compute all triple areas
        # For each triple (i,j,k), area = abs((xj-xi)*(yk-yi) - (yj-yi)*(xk-xi))
        best = 0
        counts = Counter()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - 
                            (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                    if a > 0:
                        counts[a] += 1
        if counts:
            best = max(counts.values())
        return best, counts

    # Even faster: compute all areas at once using broadcasting for pairs, then combine
    def compute_all_areas(points):
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        # areas[i,j,k] for i<j<k
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - 
                            (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                    if a > 0:
                        areas.append(a)
        return areas

    # Full histogram computation
    def full_histogram(points):
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        counts = Counter()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - 
                            (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                    if a > 0:
                        counts[a] += 1
        return counts

    def max_multiplicity(counts):
        if not counts:
            return 0
        return max(counts.values())

    # Generate initial candidates
    candidates = []
    
    # 1. Rectangular Grids
    # 5x6
    pts = [(x, y) for x in range(5) for y in range(6)]
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
    cx, cy = 0, -7
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)
    
    # 6x5
    pts = [(x, y) for x in range(6) for y in range(5)]
    cx, cy = -2, -2
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)
    
    # 2. Sheared Grids (Affine transformations with det 1)
    base = [(x, y) for x in range(5) for y in range(6)]
    for k in [1, -1, 2, -2, 3, -3]:
        pts = [(x, y + k*x) for x, y in base]
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
             xs = [p[0] for p in pts]
             ys = [p[1] for p in pts]
             cx = (min(xs) + max(xs)) // 2
             cy = (min(ys) + max(ys)) // 2
             pts = [(x - cx, y - cy) for x, y in pts]
             candidates.append(pts)

    # 3. More varied grids
    # 4x7 plus 2 extra
    pts = [(x, y) for x in range(4) for y in range(7)]
    # Add 2 more points
    pts.append((4, 3))
    pts.append((4, 4))
    cx, cy = -2, -3
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)

    # 7x4 plus 1 extra
    pts = [(x, y) for x in range(7) for y in range(4)]
    pts.append((7, 2))
    cx, cy = -3, -2
    pts = [(x + cx, y + cy) for x, y in pts]
    candidates.append(pts)

    # Hexagonal-like pattern
    hex_pts = []
    for row in range(6):
        offset = (row % 2) * 1
        for col in range(5):
            hex_pts.append((col + offset, row))
    # Remove one to get 30
    if len(hex_pts) > 30:
        hex_pts = hex_pts[:30]
    elif len(hex_pts) < 30:
        # Fill with nearby points
        existing = set(hex_pts)
        for x in range(-1, 6):
            for y in range(-1, 7):
                if (x, y) not in existing and len(hex_pts) < 30:
                    hex_pts.append((x, y))
                    existing.add((x, y))
    cx, cy = -2, -3
    hex_pts = [(x + cx, y + cy) for x, y in hex_pts]
    if len(set(hex_pts)) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in hex_pts):
        candidates.append(hex_pts)

    best_points = None
    best_score = 0
    best_hist = None
    
    # Evaluate initial candidates
    for pts in candidates:
        if len(set(pts)) != 30: 
            continue
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            continue
        counts = full_histogram(pts)
        score = max_multiplicity(counts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            best_hist = counts

    if best_points is None:
        return candidates[0]
        
    # Local Search with incremental histogram updates
    def update_histogram(points, hist, moved_idx, old_point, new_point):
        n = len(points)
        # Remove old triangles involving moved_idx
        for j in range(n):
            if j == moved_idx: 
                continue
            for k in range(j+1, n):
                if k == moved_idx: 
                    continue
                a = doubled_area(old_point, points[j], points[k])
                if a > 0:
                    if a in hist:
                        hist[a] -= 1
                        if hist[a] == 0:
                            del hist[a]
        
        # Add new triangles involving moved_idx
        for j in range(n):
            if j == moved_idx: 
                continue
            for k in range(j+1, n):
                if k == moved_idx: 
                    continue
                a = doubled_area(new_point, points[j], points[k])
                if a > 0:
                    hist[a] = hist.get(a, 0) + 1
                    
        return max_multiplicity(hist)

    def local_search(points, hist, score, time_limit, start_time):
        """Perform local search, returning improved points, hist, score."""
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            points = points[:]
            hist = hist.copy()
            current_score = score
            
            for i in range(30):
                if (time.time() - start_time) > time_limit:
                    break
                old_p = points[i]
                
                # Try neighbors in a small radius
                found_improvement = False
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0: 
                            continue
                        new_p = (old_p[0] + dx, old_p[1] + dy)
                        if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
                            continue
                        # Check distinctness
                        is_dup = False
                        for j in range(30):
                            if j != i and points[j] == new_p:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                            
                        hist_trial = hist.copy()
                        new_score = update_histogram(points, hist_trial, i, old_p, new_p)
                        
                        if new_score > current_score:
                            points[i] = new_p
                            hist = hist_trial
                            current_score = new_score
                            improved = True
                            found_improvement = True
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            score = current_score
        
        return points, hist, score

    # Run local search
    best_points, best_hist, best_score = local_search(
        best_points, best_hist, best_score, time_limit, start_time
    )

    # Targeted boost: focus on the area with highest multiplicity
    if (time.time() - start_time) < time_limit and best_hist:
        target_area = max(best_hist, key=best_hist.get)
        target_count = best_hist[target_area]
        
        # Try to increase count of target_area by moving points
        points = best_points[:]
        hist = best_hist.copy()
        current_score = best_score
        
        # For each point, try positions that might create more triangles with target_area
        for i in range(30):
            if (time.time() - start_time) > time_limit:
                break
            old_p = points[i]
            
            # Try a wider range of positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0: 
                        continue
                    new_p = (old_p[0] + dx, old_p[1] + dy)
                    if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
                        continue
                    is_dup = False
                    for j in range(30):
                        if j != i and points[j] == new_p:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    hist_trial = hist.copy()
                    new_score = update_histogram(points, hist_trial, i, old_p, new_p)
                    
                    # Accept if it increases the target area count or overall score
                    new_target_count = hist_trial.get(target_area, 0)
                    if new_score > current_score or (new_score == current_score and new_target_count > hist.get(target_area, 0)):
                        points[i] = new_p
                        hist = hist_trial
                        current_score = new_score
                        if current_score > best_score:
                            best_points = points[:]
                            best_hist = hist
                            best_score = current_score
                            # Update target if it changed
                            if hist:
                                new_target = max(hist, key=hist.get)
                                if new_target != target_area:
                                    target_area = new_target
                                    target_count = hist[target_area]
                        break

        if current_score > best_score:
            best_points = points
            best_hist = hist
            best_score = current_score

    # Final polish: try larger moves
    if (time.time() - start_time) < time_limit:
        points = best_points[:]
        hist = best_hist.copy()
        current_score = best_score
        
        for i in range(30):
            if (time.time() - start_time) > time_limit:
                break
            old_p = points[i]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0: 
                        continue
                    new_p = (old_p[0] + dx, old_p[1] + dy)
                    if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
                        continue
                    is_dup = False
                    for j in range(30):
                        if j != i and points[j] == new_p:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    hist_trial = hist.copy()
                    new_score = update_histogram(points, hist_trial, i, old_p, new_p)
                    
                    if new_score > current_score:
                        points[i] = new_p
                        hist = hist_trial
                        current_score = new_score
                        best_points = points[:]
                        best_hist = hist
                        best_score = current_score
                        break

    return best_points
