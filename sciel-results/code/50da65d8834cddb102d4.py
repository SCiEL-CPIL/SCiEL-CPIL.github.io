"""Optimized search for 30 points maximizing equal-area triangle multiplicity."""
import numpy as np
from collections import Counter
import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # safety margin before 60s limit
    
    def compute_max_multiplicity(points):
        """Fast vectorized computation of max area multiplicity."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Precompute all pairwise differences
        # For triple (i,j,k): area2 = |xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj)|
        # = |(xi-xk)*(yj-yk) - (xj-xk)*(yi-yk)|
        
        # Use the fact that area2(i,j,k) = |(xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)|
        # For efficiency, compute all C(30,3) = 4060 areas
        
        areas = np.zeros(4060, dtype=np.int64)
        idx = 0
        for i in range(n - 2):
            xi, yi = x[i], y[i]
            for j in range(i + 1, n - 1):
                xj, yj = x[j], y[j]
                dx1, dy1 = xj - xi, yj - yi
                for k in range(j + 1, n):
                    areas[idx] = abs(dx1 * (y[k] - yi) - dy1 * (x[k] - xi))
                    idx += 1
        
        areas = areas[areas > 0]
        if len(areas) == 0:
            return 0
        
        values, counts = np.unique(areas, return_counts=True)
        return int(np.max(counts))
    
    def get_area_counts(points):
        """Return dict of area -> count."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        
        areas = np.zeros(4060, dtype=np.int64)
        idx = 0
        for i in range(n - 2):
            xi, yi = x[i], y[i]
            for j in range(i + 1, n - 1):
                xj, yj = x[j], y[j]
                dx1, dy1 = xj - xi, yj - yi
                for k in range(j + 1, n):
                    areas[idx] = abs(dx1 * (y[k] - yi) - dy1 * (x[k] - xi))
                    idx += 1
        
        areas = areas[areas > 0]
        if len(areas) == 0:
            return {}
        
        values, counts = np.unique(areas, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}
    
    # Generate structured candidates
    candidates = []
    
    # Two-row configurations with various spacings
    for n1 in range(10, 21):
        n2 = 30 - n1
        if n2 < 5:
            continue
        for x_start in range(-25, 26 - n1):
            for gap in range(1, 4):
                row1 = [(x_start + i * gap, 0) for i in range(n1)]
                row2 = [(x_start + i * gap, gap) for i in range(n2)]
                if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in row1 + row2) and len(set(row1 + row2)) == 30:
                    candidates.append(row1 + row2)
    
    # Three-row configurations
    for n1 in range(8, 14):
        for n2 in range(8, 14):
            n3 = 30 - n1 - n2
            if n3 < 5:
                continue
            for x_start in range(-20, 21 - max(n1, n2, n3)):
                for gap in [1, 2]:
                    row1 = [(x_start + i, 0) for i in range(n1)]
                    row2 = [(x_start + i, gap) for i in range(n2)]
                    row3 = [(x_start + i, 2 * gap) for i in range(n3)]
                    pts = row1 + row2 + row3
                    if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Evaluate all structured candidates
    random.seed(42)
    best_score = 0
    best_points = None
    
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        score = compute_max_multiplicity(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]
    
    # If no good structured candidate, use a default
    if best_points is None:
        best_points = [(i, 0) for i in range(15)] + [(i, 1) for i in range(15)]
        best_score = compute_max_multiplicity(best_points)
    
    # Local search with neighborhood moves
    all_valid = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    for iteration in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        current_set = set(best_points)
        
        for i in range(30):
            if time.time() - start_time > DEADLINE:
                break
            
            old_point = best_points[i]
            tried = set()
            
            # Try nearby moves first (more likely to be good)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = old_point[0] + dx, old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ not in current_set and np_ != old_point and np_ not in tried:
                            tried.add(np_)
                            new_points = best_points[:]
                            new_points[i] = np_
                            score = compute_max_multiplicity(new_points)
                            if score > best_score:
                                best_score = score
                                best_points = new_points
                                current_set = set(best_points)
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
            
            # If no nearby improvement, try some random moves
            for _ in range(10):
                if time.time() - start_time > DEADLINE:
                    break
                np_ = random.choice(all_valid)
                if np_ not in current_set and np_ != old_point and np_ not in tried:
                    tried.add(np_)
                    new_points = best_points[:]
                    new_points[i] = np_
                    score = compute_max_multiplicity(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        current_set = set(best_points)
                        improved = True
                        break
        
        if not improved:
            # Try larger perturbations
            new_points = best_points[:]
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, 29)
                old = new_points[idx]
                for _ in range(100):
                    dx = random.randint(-10, 10)
                    dy = random.randint(-10, 10)
                    np_ = (old[0] + dx, old[1] + dy)
                    if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in new_points:
                        new_points[idx] = np_
                        break
            
            if len(set(new_points)) == 30:
                score = compute_max_multiplicity(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    improved = True
    
    return best_points
