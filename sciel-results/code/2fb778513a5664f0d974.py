import time
import random
import numpy as np
from collections import Counter

def _make_grid(cols, rows, x0=0, y0=0):
    """Generate a cols x rows grid of points starting at (x0, y0)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c, y0 + r))
    return pts

def _make_sheared_grid(cols, rows, s, x0=0, y0=0):
    """
    Generate a sheared grid.
    Basis vectors: (1, 0) and (s, 1).
    Points: (x0 + c + s*r, y0 + r) for c in [0, cols), r in [0, rows).
    """
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c + s * r
            y = y0 + r
            pts.append((x, y))
    return pts

def _full_score_and_target(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    counts = Counter()
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                if d > 0:
                    counts[d] += 1
    if not counts:
        return 0, 0
    target_area, max_count = counts.most_common(1)[0]
    return max_count, target_area

def _delta_count_for_target(points, idx, new_pt, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx] is replaced by new_pt.
    """
    old_pt = points[idx]
    others = [points[j] for j in range(len(points)) if j != idx]
    
    delta = 0
    n_others = len(others)
    
    for j in range(n_others):
        xj, yj = others[j]
        for k in range(j+1, n_others):
            xk, yk = others[k]
            
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
                
    return delta

def _delta_count_for_swap(points, idx1, idx2, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx1] and points[idx2] are swapped.
    """
    n = len(points)
    p1 = points[idx1]
    p2 = points[idx2]
    delta = 0
    
    other_indices = [i for i in range(n) if i != idx1 and i != idx2]
    
    # Case 1: (idx1, j, k) -> (idx2, j, k)
    for a in range(len(other_indices)):
        j = other_indices[a]
        xj, yj = points[j]
        for b in range(a+1, len(other_indices)):
            k = other_indices[b]
            xk, yk = points[k]
            
            d_old = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            d_new = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
    
    # Case 2: (idx2, j, k) -> (idx1, j, k)
    for a in range(len(other_indices)):
        j = other_indices[a]
        xj, yj = points[j]
        for b in range(a+1, len(other_indices)):
            k = other_indices[b]
            xk, yk = points[k]
            
            d_old = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            d_new = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
    
    # Case 3: (idx1, idx2, j) -> (idx2, idx1, j)
    for j in other_indices:
        xj, yj = points[j]
        
        d_old = abs((p2[0] - p1[0]) * (yj - p1[1]) - (p2[1] - p1[1]) * (xj - p1[0]))
        d_new = abs((p1[0] - p2[0]) * (yj - p2[1]) - (p1[1] - p2[1]) * (xj - p2[0]))
        
        if d_old == target_area and d_old > 0:
            delta -= 1
        if d_new == target_area and d_new > 0:
            delta += 1
    
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    # 1. Generate diverse initial candidates
    candidates = []
    
    # Rectangular grids
    for cols, rows in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
        x0 = -cols // 2
        y0 = -rows // 2
        pts = _make_grid(cols, rows, x0, y0)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            candidates.append(pts)
            
    # Sheared grids
    for cols, rows in [(6, 5), (5, 6), (10, 3), (3, 10)]:
        for s in [-2, -1, 1, 2]:
            x0 = -cols // 2 - s * (rows // 2)
            y0 = -rows // 2
            pts = _make_sheared_grid(cols, rows, s, x0, y0)
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                candidates.append(pts)

    # 2. Evaluate all candidates, pick the best
    best_pts = None
    best_count = -1
    target_area = 1
    
    for pts in candidates:
        if len(set(pts)) != 30:
            continue
        c, t = _full_score_and_target(pts)
        if c > best_count:
            best_count = c
            target_area = t
            best_pts = [p for p in pts]

    if best_pts is None:
        best_pts = _make_grid(6, 5, -2, -2)
        best_count, target_area = _full_score_and_target(best_pts)

    # 3. Simulated Annealing
    random.seed(42)
    
    current_pts = best_pts[:]
    current_count = best_count
    current_target = target_area
    
    T = 10.0
    cooling = 0.9997
    min_T = 0.01
    last_reval_time = start_time
    
    while time.time() < deadline:
        # Periodic re-evaluation (every ~2 seconds)
        if time.time() - last_reval_time > 2.0:
            c, t = _full_score_and_target(current_pts)
            if c > best_count:
                best_count = c
                best_pts = current_pts[:]
            # Update target to the current max area
            current_target = t
            current_count = c
            last_reval_time = time.time()
            T = max(T, 5.0)  # Reheat on re-evaluation
        
        # Propose a move: 40% small move, 30% large jump, 30% swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Small move: shift one point by a small vector
            idx = random.randint(0, 29)
            orig = current_pts[idx]
            radius = 3
            candidates_move = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in current_pts:
                            candidates_move.append((nx, ny))
            
            if not candidates_move:
                continue
            new_pt = random.choice(candidates_move)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            # Acceptance criterion
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
                    
        elif move_type < 0.7:
            # Large jump: move one point to a random location
            idx = random.randint(0, 29)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) in current_pts:
                continue
            new_pt = (nx, ny)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
                    
        else:
            # Swap two points
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 == idx2:
                continue
            delta = _delta_count_for_swap(current_pts, idx1, idx2, current_target)
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx1], current_pts[idx2] = current_pts[idx2], current_pts[idx1]
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            # Reheat if stuck
            T = 5.0

    # 4. Final deterministic polish: try all single-point moves with small radius
    if time.time() < deadline - 2:
        for iteration in range(20):
            if time.time() > deadline - 1:
                break
            improved_any = False
            for idx in range(30):
                orig = best_pts[idx]
                existing = set(best_pts)
                best_delta = 0
                best_new_pt = None
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig[0] + dx, orig[1] + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        if (nx, ny) in existing:
                            continue
                        delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                        if delta > best_delta:
                            best_delta = delta
                            best_new_pt = (nx, ny)
                if best_new_pt and best_delta > 0:
                    best_pts[idx] = best_new_pt
                    improved_any = True
            if not improved_any:
                break
        
        # Update best after polish
        c, t = _full_score_and_target(best_pts)
        if c > best_count:
            best_count = c
            target_area = t

    return [list(p) for p in best_pts]
