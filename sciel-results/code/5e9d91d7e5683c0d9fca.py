"""Combined approach: structured lattice initialization + fast local search + targeted area optimization."""
import numpy as np
from collections import Counter
import random
import time

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Start with structured lattice patterns (two-row parallel lines are strong).
    2. Use fast area computation with vectorized ops.
    3. Apply enhanced local search with structure-aware move generation.
    4. Targeted optimization: try to increase count for the current best area.
    5. Deadline guard: return best found before 60s wall-clock.
    """
    
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s limit
    
    def compute_max_multiplicity(points):
        """Compute the maximum multiplicity of any positive doubled area using vectorized ops."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Precompute all C(n,3) triangle areas
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
                    if area > 0:
                        areas.append(area)
        
        if not areas:
            return 0, 0, 0
            
        arr = np.array(areas, dtype=np.int64)
        values, counts = np.unique(arr, return_counts=True)
        max_idx = np.argmax(counts)
        return int(counts[max_idx]), int(values[max_idx]), len(arr)
    
    def count_area_multiplicities(points):
        """Return dict: area -> count for all positive areas."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
                    if area > 0:
                        areas.append(area)
        
        if not areas:
            return {}
            
        arr = np.array(areas, dtype=np.int64)
        values, counts = np.unique(arr, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}
    
    def count_specific_area(points, target_area):
        """Count triangles with exactly target_area."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
                    if area == target_area:
                        count += 1
        return count
    
    # Generate diverse candidate sets from structured patterns
    candidates = []
    
    # Pattern A: Two parallel rows (strong from inspiration)
    for x_start in range(-15, 16):
        for y0 in range(-15, 16):
            for y1 in [y0+1, y0+2, y0+3, y0+4, y0+5]:
                pts = []
                for x in range(x_start, x_start + 15):
                    if -30 <= x <= 30:
                        pts.append((x, y0))
                for x in range(x_start, x_start + 15):
                    if -30 <= x <= 30:
                        pts.append((x, y1))
                if len(pts) == 30 and len(set(pts)) == 30:
                    candidates.append(pts[:])
    
    # Pattern B: Rectangular grids
    for w in range(4, 10):
        for h in range(4, 10):
            if w * h >= 30:
                for ox in range(-15, 15):
                    for oy in range(-15, 15):
                        pts = [(ox + x, oy + y) for x in range(w) for y in range(h)]
                        if len(pts) >= 30:
                            pts = pts[:30]
                            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                                candidates.append(pts)
    
    # Pattern C: Three parallel rows
    for x_start in range(-10, 10):
        for base_y in range(-10, 10):
            for gap in [1, 2, 3]:
                pts = []
                for row in range(3):
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, base_y + row * gap))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Pattern D: Four parallel rows
    for x_start in range(-10, 10):
        for base_y in range(-10, 10):
            for gap in [1, 2]:
                pts = []
                for row in range(4):
                    for x in range(x_start, x_start + 8):
                        if -30 <= x <= 30:
                            pts.append((x, base_y + row * gap))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Pattern E: Diagonal lines
    for offset in range(-10, 11):
        for slope in [1, 2, 3]:
            pts = []
            for t in range(-15, 16):
                x = t
                y = slope * t + offset
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
            if len(pts) >= 30:
                pts = pts[:30]
                if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                    candidates.append(pts)
    
    # Evaluate all candidates
    random.seed(42)
    best_score = 0
    best_points = None
    best_area = None
    
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        score, area, total = compute_max_multiplicity(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]
            best_area = area
    
    if best_points is None:
        # Fallback: two-row configuration
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score, best_area, _ = compute_max_multiplicity(best_points)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_points
    
    # Enhanced local search with structure-aware moves
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Phase 1: General local search
    for iteration in range(15):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        current_set = set(best_points)
        
        for i in range(30):
            if time.time() - start_time > DEADLINE:
                break
            
            old_point = best_points[i]
            
            # Generate candidate replacements
            candidates_to_try = []
            
            # Points in a neighborhood
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ not in current_set and np_ != old_point:
                            candidates_to_try.append(np_)
            
            # Also try some random points
            for _ in range(10):
                np_ = random.choice(all_valid_points)
                if np_ not in current_set and np_ != old_point:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:20]:
                if time.time() - start_time > DEADLINE:
                    break
                
                new_points = best_points[:]
                new_points[i] = new_point
                
                score, area, _ = compute_max_multiplicity(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    best_area = area
                    current_set = set(best_points)
                    improved = True
                    break
        
        if not improved:
            # Try a few random restarts within the neighborhood
            for _ in range(3):
                if time.time() - start_time > DEADLINE:
                    break
                
                new_points = best_points[:]
                for _ in range(random.randint(2, 4)):
                    idx = random.randint(0, 29)
                    old = new_points[idx]
                    for _ in range(50):
                        dx = random.randint(-5, 5)
                        dy = random.randint(-5, 5)
                        np_ = (old[0] + dx, old[1] + dy)
                        if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in new_points:
                            new_points[idx] = np_
                            break
                
                if len(set(new_points)) == 30:
                    score, area, _ = compute_max_multiplicity(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        best_area = area
                        improved = True
    
    if time.time() - start_time > DEADLINE:
        return best_points
    
    # Phase 2: Targeted optimization - try to increase count for best_area
    # This focuses the search on improving the specific area that's currently most frequent
    current_counts = count_area_multiplicities(best_points)
    target_area = best_area
    target_count = current_counts.get(target_area, 0)
    
    for iteration in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        current_set = set(best_points)
        
        for i in range(30):
            if time.time() - start_time > DEADLINE:
                break
            
            old_point = best_points[i]
            
            # Generate candidate replacements
            candidates_to_try = []
            
            # Points in a larger neighborhood for targeted search
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ not in current_set and np_ != old_point:
                            candidates_to_try.append(np_)
            
            # Random points
            for _ in range(15):
                np_ = random.choice(all_valid_points)
                if np_ not in current_set and np_ != old_point:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:30]:
                if time.time() - start_time > DEADLINE:
                    break
                
                new_points = best_points[:]
                new_points[i] = new_point
                
                # Check if this improves the target area count
                new_target_count = count_specific_area(new_points, target_area)
                
                if new_target_count > target_count:
                    # Verify overall score
                    score, area, _ = compute_max_multiplicity(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        best_area = area
                        target_count = count_specific_area(best_points, target_area)
                        current_set = set(best_points)
                        improved = True
                        break
                    elif score == best_score and area == target_area:
                        # Same max score but improved target count - accept for diversity
                        best_points = new_points
                        target_count = new_target_count
                        current_set = set(best_points)
                        improved = True
                        break
        
        if not improved:
            # Try perturbing multiple points
            for _ in range(3):
                if time.time() - start_time > DEADLINE:
                    break
                
                new_points = best_points[:]
                for _ in range(random.randint(2, 3)):
                    idx = random.randint(0, 29)
                    old = new_points[idx]
                    for _ in range(50):
                        dx = random.randint(-8, 8)
                        dy = random.randint(-8, 8)
                        np_ = (old[0] + dx, old[1] + dy)
                        if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in new_points:
                            new_points[idx] = np_
                            break
                
                if len(set(new_points)) == 30:
                    new_target_count = count_specific_area(new_points, target_area)
                    if new_target_count > target_count:
                        score, area, _ = compute_max_multiplicity(new_points)
                        if score >= best_score:
                            best_points = new_points
                            if score > best_score:
                                best_score = score
                                best_area = area
                            target_count = new_target_count
                            improved = True
    
    return best_points
