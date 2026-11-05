import time
import random
import numpy as np
from collections import Counter

def _full_score_numpy(points):
    """
    Compute the histogram of doubled areas using NumPy.
    Returns (max_count, target_area).
    """
    pts = np.array(points)
    n = len(pts)
    if n < 3:
        return 0, 0
    
    # Generate all unique triples i < j < k
    # This is memory intensive for large n, but n=30 is small.
    # C(30, 3) = 4060 triples.
    # We can iterate or use broadcasting. Broadcasting is cleaner.
    
    # Create indices for all triples
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_arr = np.array(i_idx)
    j_arr = np.array(j_idx)
    k_arr = np.array(k_idx)
    
    vi = pts[i_arr]
    vj = pts[j_arr]
    vk = pts[k_arr]
    
    # Vectors from i to j and i to k
    v_ji = vj - vi
    v_ki = vk - vi
    
    # Determinant: x1*y2 - y1*x2
    det = v_ji[:, 0] * v_ki[:, 1] - v_ji[:, 1] * v_ki[:, 0]
    areas = np.abs(det)
    
    # Exclude zero area
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0, 0
        
    # Find the most common area
    unique_areas, counts = np.unique(areas, return_counts=True)
    max_idx = np.argmax(counts)
    return int(counts[max_idx]), int(unique_areas[max_idx])

def _delta_count_for_target(points, idx, new_pt, target_area):
    """
    Calculate change in count of triangles with target_area when points[idx] is replaced.
    points is a list of tuples.
    """
    n = len(points)
    old_pt = points[idx]
    delta = 0
    # Iterate over all pairs (j, k) where j != idx and k != idx
    # To avoid double counting and self-interaction, we just iterate all j, k pairs
    # and check if they involve idx.
    # Actually, the triangles involving idx are (idx, j, k) for all j < k, j!=idx, k!=idx.
    # There are C(29, 2) = 406 such triangles.
    for j in range(n):
        if j == idx: continue
        for k in range(j+1, n):
            if k == idx: continue
            xj, yj = points[j]
            xk, yk = points[k]
            
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 5s margin
    
    best_count = 0
    best_pts = None
    best_target = 0
    
    # Generate initial candidates
    candidates = []
    
    # 1. Rectangular Grids
    for cols in [2, 3, 5, 6, 10, 15]:
        rows = 30 // cols
        if rows * cols != 30: continue
        for x0 in range(-10, 11, 2):
            for y0 in range(-10, 11, 2):
                pts = [(x0 + c, y0 + r) for r in range(rows) for c in range(cols)]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
                    
    # 2. Two Parallel Lines
    for gap in range(1, 10):
        for x0 in range(-10, 11, 2):
            for y0 in range(-10, 11, 2):
                pts = [(x0 + i, y0) for i in range(15)] + [(x0 + i, y0 + gap) for i in range(15)]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
                    
    # 3. Three Parallel Lines
    for gap in range(1, 8):
        for x0 in range(-10, 11, 2):
            for y0 in range(-10, 11, 2):
                pts = ([(x0 + i, y0) for i in range(10)] +
                       [(x0 + i, y0 + gap) for i in range(10)] +
                       [(x0 + i, y0 + 2*gap) for i in range(10)])
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)

    # 4. Sheared Grids (Parallelograms)
    for cols in [5, 6]:
        rows = 30 // cols
        for shear in range(-5, 6):
            for x0 in range(-10, 11, 2):
                for y0 in range(-10, 11, 2):
                    pts = []
                    valid = True
                    for r in range(rows):
                        for c in range(cols):
                            x = x0 + c + shear * r
                            y = y0 + r
                            if not (-30 <= x <= 30 and -30 <= y <= 30):
                                valid = False
                                break
                            pts.append((x, y))
                        if not valid: break
                    if valid:
                        candidates.append(pts)

    # 5. Random Initializations
    random.seed(42)
    for _ in range(50):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))

    # Evaluate candidates and run local search on the best ones
    # Sort candidates by initial score to focus search on promising ones
    scored_candidates = []
    for pts in candidates:
        c, t = _full_score_numpy(pts)
        if c > best_count:
            best_count = c
            best_pts = list(pts)
            best_target = t
        scored_candidates.append((c, t, pts))
        
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Run local search on top 10 candidates
    for c, t, init_pts in scored_candidates[:10]:
        if time.time() > deadline:
            break
            
        current_pts = list(init_pts)
        current_count = c
        current_target = t
        no_improve = 0
        steps = 0
        
        while time.time() < deadline:
            steps += 1
            
            # Periodically re-evaluate target to escape local optima of a specific area
            if no_improve > 100:
                new_c, new_t = _full_score_numpy(current_pts)
                if new_c > current_count:
                    current_count = new_c
                    current_target = new_t
                    no_improve = 0
                else:
                    # Allow random jump to escape
                    if random.random() < 0.2:
                        idx = random.randint(0, 29)
                        nx = random.randint(-30, 30)
                        ny = random.randint(-30, 30)
                        if (nx, ny) not in current_pts:
                            current_pts[idx] = (nx, ny)
                            no_improve = 0
                            continue
                    no_improve = 0
                    # If still no improvement after re-eval and jump, try smaller radius or break?
                    # Let's just reset no_improve and continue, but maybe increase jump probability next time
                    if no_improve > 200:
                        break
                    continue
            
            idx = random.randint(0, 29)
            orig = current_pts[idx]
            improved = False
            
            # Try small moves first
            radius = 2
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    if (nx, ny) in current_pts:
                        continue
                    delta = _delta_count_for_target(current_pts, idx, (nx, ny), current_target)
                    if delta > 0:
                        current_pts[idx] = (nx, ny)
                        current_count += delta
                        no_improve = 0
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                
                # If stuck, try a random jump
                if no_improve > 50 and random.random() < 0.1:
                    idx = random.randint(0, 29)
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    if (nx, ny) not in current_pts:
                        delta = _delta_count_for_target(current_pts, idx, (nx, ny), current_target)
                        # Accept if it improves, or if it's neutral and we're stuck
                        if delta > 0 or (delta == 0 and random.random() < 0.1):
                            current_pts[idx] = (nx, ny)
                            current_count += delta
                            no_improve = 0
            
            # Update global best
            if current_count > best_count:
                best_count = current_count
                best_pts = list(current_pts)
                best_target = current_target

    return [list(p) for p in best_pts]
