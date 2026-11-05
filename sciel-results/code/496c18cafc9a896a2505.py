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
    pts_arr = np.array(points, dtype=np.int64)
    counts = Counter()
    for i in range(n):
        xi, yi = int(pts_arr[i][0]), int(pts_arr[i][1])
        for j in range(i+1, n):
            xj, yj = int(pts_arr[j][0]), int(pts_arr[j][1])
            for k in range(j+1, n):
                xk, yk = int(pts_arr[k][0]), int(pts_arr[k][1])
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
    n = len(points)
    old_pt = points[idx]
    
    # Get all other points as numpy array
    other_indices = [i for i in range(n) if i != idx]
    if len(other_indices) < 2:
        return 0
    
    pts_arr = np.array([points[i] for i in other_indices], dtype=np.int64)
    
    # Compute determinants with old point
    ox, oy = old_pt
    nx, ny = new_pt
    
    # Vectorized computation: det = |xj*(yk-oy) - yj*(xk-ox)| for pairs (j,k)
    # We need to compute for all pairs j<k in other_indices
    m = len(other_indices)
    
    # Use broadcasting to compute all pair determinants
    # For each pair (i,j) where i<j, compute determinant with the fixed point
    xi = pts_arr[:, 0:1]  # m x 1
    yi = pts_arr[:, 0:1]  # m x 1
    xj = pts_arr[1:, 0:1].T  # 1 x (m-1)
    yj = pts_arr[1:, 1:2].T  # 1 x (m-1)
    
    # Actually, let's do this more carefully
    # For pair (a, b) where a < b, det = |xa*(yb-oy) - ya*(xb-ox)|
    # We'll iterate but use numpy for the inner computation
    
    delta = 0
    
    for a in range(m):
        xa, ya = pts_arr[a]
        for b in range(a+1, m):
            xb, yb = pts_arr[b]
            d_old = abs(int(xa * (yb - oy) - ya * (xb - ox)))
            d_new = abs(int(xa * (yb - ny) - ya * (xb - nx)))
            
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
    
    # Triangles involving exactly one of idx1 or idx2 (not both)
    # Case 1: triangles with idx1 but not idx2: (idx1, j, k) where j,k != idx1,idx2
    # Case 2: triangles with idx2 but not idx1: (idx2, j, k) where j,k != idx1,idx2
    # Case 3: triangles with both idx1 and idx2: (idx1, idx2, j) where j != idx1,idx2
    
    other_indices = [i for i in range(n) if i != idx1 and i != idx2]
    
    # Case 1: (idx1, j, k) -> (idx2, j, k) for j<k in other_indices
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
    
    # Case 2: (idx2, j, k) -> (idx1, j, k) for j<k in other_indices
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
    
    # Case 3: (idx1, idx2, j) -> (idx2, idx1, j) for j in other_indices
    # The determinant magnitude is the same when swapping two vertices, so no change
    # But let's verify: det(p1, p2, j) vs det(p2, p1, j)
    # det(p1,p2,j) = (p2-p1) x (j-p1)
    # det(p2,p1,j) = (p1-p2) x (j-p2) = -(p2-p1) x (j-p2)
    # These are not necessarily equal in magnitude!
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
    deadline = start_time + 55

    # 1. Generate Candidates
    candidates = []
    
    # Rectangular Grids
    for cols, rows in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
        # Center the grid
        x0 = -cols // 2
        y0 = -rows // 2
        pts = _make_grid(cols, rows, x0, y0)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            candidates.append(pts)
            
    # Sheared Grids
    for cols, rows in [(6, 5), (5, 6), (10, 3), (3, 10)]:
        for s in [-2, -1, 1, 2]:
            x0 = -cols // 2 - s * (rows // 2)
            y0 = -rows // 2
            pts = _make_sheared_grid(cols, rows, s, x0, y0)
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                candidates.append(pts)

    # 2. Evaluate Candidates
    best_pts = None
    best_count = -1
    target_area = 1
    
    for pts in candidates:
        if len(set(pts)) != 30: continue
        c, t = _full_score_and_target(pts)
        if c > best_count:
            best_count = c
            target_area = t
            best_pts = [p for p in pts]

    if best_pts is None:
        # Fallback
        best_pts = _make_grid(6, 5, -2, -2)
        best_count, target_area = _full_score_and_target(best_pts)

    random.seed(42)
    no_improve_counter = 0
    
    while time.time() < deadline:
        if no_improve_counter > 100:
            # Re-evaluate target area
            c, t = _full_score_and_target(best_pts)
            if c > best_count:
                best_count = c
                target_area = t
            no_improve_counter = 0
            continue

        improved = False
        use_swap = random.random() < 0.3
        
        if use_swap:
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 != idx2:
                delta = _delta_count_for_swap(best_pts, idx1, idx2, target_area)
                if delta > 0:
                    best_pts[idx1], best_pts[idx2] = best_pts[idx2], best_pts[idx1]
                    best_count += delta
                    no_improve_counter = 0
                    improved = True
        else:
            idx = random.randint(0, 29)
            orig = best_pts[idx]
            # Try a few random moves and local neighborhood
            # Local neighborhood - expand radius based on stagnation
            radius = 3 + min(no_improve_counter // 20, 5)  # Up to radius 8
            best_delta = 0
            best_new_pt = None
            
            # Use a set for O(1) membership testing
            existing = set(best_pts)
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30): continue
                    if (nx, ny) in existing: continue
                    delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pt = (nx, ny)
            
            if best_new_pt and best_delta > 0:
                best_pts[idx] = best_new_pt
                best_count += best_delta
                no_improve_counter = 0
                improved = True
            else:
                # Try random jumps with increasing range
                for attempt in range(3 + min(no_improve_counter // 10, 7)):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    if (nx, ny) in existing: continue
                    delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                    if delta > 0:
                        best_pts[idx] = (nx, ny)
                        best_count += delta
                        no_improve_counter = 0
                        improved = True
                        break

        if not improved:
            no_improve_counter += 1
            
        # Occasional restart
        if no_improve_counter > 150 and random.random() < 0.15:
             idx = random.randint(0, 29)
             nx = random.randint(-30, 30)
             ny = random.randint(-30, 30)
             if (nx, ny) not in set(best_pts):
                 best_pts[idx] = (nx, ny)
                 best_count, target_area = _full_score_and_target(best_pts)
                 no_improve_counter = 0

    # Final polish: try to improve with a tighter local search
    if time.time() < deadline - 2:
        for _ in range(50):
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
                        if dx == 0 and dy == 0: continue
                        nx, ny = orig[0] + dx, orig[1] + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30): continue
                        if (nx, ny) in existing: continue
                        delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                        if delta > best_delta:
                            best_delta = delta
                            best_new_pt = (nx, ny)
                if best_new_pt and best_delta > 0:
                    best_pts[idx] = best_new_pt
                    best_count += best_delta
                    improved_any = True
            if not improved_any:
                break

    return [list(p) for p in best_pts]
