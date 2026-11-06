"""Combined approach: structured lattice initialization + fast local search."""
import numpy as np
from collections import Counter
import random
import time
from itertools import combinations

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Start with structured lattice patterns (two-row parallel lines are strong).
    2. Use fast area computation with vectorized ops.
    3. Apply enhanced local search with structure-aware move generation.
    4. Deadline guard: return best found before 60s wall-clock.
    """
    
    start_time = time.time()
    DEADLINE = 58.0  # safety margin before 60s limit
    
    def compute_max_multiplicity(points):
        """Compute the maximum multiplicity of any positive doubled area using vectorized ops."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        
        # Generate all triples
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    xi, yi = pts[i]
                    xj, yj = pts[j]
                    xk, yk = pts[k]
                    area = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if area > 0:
                        areas.append(area)
        
        if not areas:
            return 0
        
        counter = Counter(areas)
        return max(counter.values())
    
    def fast_score(points, target_area=None):
        """
        Fast scoring: if target_area is given, count triangles with that area.
        Otherwise, compute full max multiplicity.
        """
        if target_area is not None:
            n = len(points)
            pts = np.array(points, dtype=np.int64)
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        xi, yi = pts[i]
                        xj, yj = pts[j]
                        xk, yk = pts[k]
                        area = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                        if area == target_area:
                            count += 1
            return count
        else:
            return compute_max_multiplicity(points)
    
    def count_area_multiplicities(points):
        """Return dict: area -> count for all positive areas."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        counts = Counter()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    xi, yi = pts[i]
                    xj, yj = pts[j]
                    xk, yk = pts[k]
                    area = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if area > 0:
                        counts[area] += 1
        return counts
    
    # Generate diverse candidate sets from structured patterns
    candidates = []
    
    # Pattern A: Two parallel rows (strong from inspiration 2)
    for x_start in range(-10, 15):
        for y0 in range(-10, 15):
            for y1 in [y0+1, y0+2, y0+3]:
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
    for w in range(4, 9):
        for h in range(4, 9):
            if w * h >= 30:
                for ox in range(-10, 10):
                    for oy in range(-10, 10):
                        pts = [(ox + x, oy + y) for x in range(w) for y in range(h)]
                        if len(pts) >= 30:
                            pts = pts[:30]
                            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                                candidates.append(pts)
    
    # Pattern C: Three parallel rows
    for x_start in range(-10, 10):
        for base_y in range(-10, 10):
            for gap in [1, 2]:
                pts = []
                for row in range(3):
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, base_y + row * gap))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Pattern D: Hexagonal-like lattice
    for scale in [1, 2, 3]:
        for ox in range(-10, 10):
            for oy in range(-10, 10):
                pts = []
                for row in range(6):
                    for col in range(5):
                        x = ox + col * 2 * scale + (row % 2) * scale
                        y = oy + row * scale
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
        score = compute_max_multiplicity(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]
            # Find which area achieved this
            counts = count_area_multiplicities(cand)
            best_area = max(counts, key=counts.get)
    
    if best_points is None:
        # Fallback: two-row configuration
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score = compute_max_multiplicity(best_points)
        counts = count_area_multiplicities(best_points)
        best_area = max(counts, key=counts.get)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_points
    
    # Enhanced local search with structure-aware moves
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Precompute area counts for current best
    current_counts = count_area_multiplicities(best_points)
    current_best_area = max(current_counts, key=current_counts.get)
    current_best_count = current_counts[current_best_area]
    
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
            
            # Points in a neighborhood
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ not in current_set and np_ != old_point:
                            candidates_to_try.append(np_)
            
            # Also try some random points
            for _ in range(20):
                np_ = random.choice(all_valid_points)
                if np_ not in current_set and np_ != old_point:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:25]:  # Limit tries per position
                if time.time() - start_time > DEADLINE:
                    break
                
                new_points = best_points[:]
                new_points[i] = new_point
                
                # Fast check: compute score
                score = compute_max_multiplicity(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    current_set = set(best_points)
                    improved = True
                    # Update area tracking
                    current_counts = count_area_multiplicities(best_points)
                    current_best_area = max(current_counts, key=current_counts.get)
                    current_best_count = current_counts[current_best_area]
                    break
        
        if not improved:
            # Try a few random restarts within the neighborhood
            for _ in range(5):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Perturb 2-3 random points
                new_points = best_points[:]
                for _ in range(random.randint(2, 3)):
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
                    score = compute_max_multiplicity(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        current_set = set(best_points)
                        improved = True
                        current_counts = count_area_multiplicities(best_points)
                        current_best_area = max(current_counts, key=current_counts.get)
                        current_best_count = current_counts[current_best_area]
    
    return best_points
