import numpy as np
import random
import time
from collections import Counter

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    """
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin (90s kill, want ~35s buffer)
    
    def compute_all_areas(pts):
        """
        Compute all triangle areas for a set of points.
        Returns:
            counts: Counter of areas
            best_area: area with max count
            best_count: max count
        """
        n = len(pts)
        if n < 3:
            return Counter(), 0, 0
            
        pts_arr = np.array(pts, dtype=np.int64)
        counts = Counter()
        
        for i in range(n):
            pi = pts_arr[i]
            others = pts_arr[i+1:]
            if len(others) < 2:
                continue
            n_o = len(others)
            j_idx, k_idx = np.triu_indices(n_o, k=1)
            xj = others[j_idx, 0]
            yj = others[j_idx, 1]
            xk = others[k_idx, 0]
            yk = others[k_idx, 1]
            
            xi, yi = pi
            areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
            non_zero_areas = areas[areas > 0]
            if len(non_zero_areas) > 0:
                counts.update(non_zero_areas.tolist())
                
        if not counts:
            return Counter(), 0, 0
            
        best_area, best_count = counts.most_common(1)[0]
        return counts, int(best_area), int(best_count)

    def compute_areas_for_point(point, others_arr):
        """Compute areas of triangles formed by point and all pairs in others_arr."""
        x0, y0 = point
        n_o = len(others_arr)
        if n_o < 2:
            return np.array([], dtype=np.int64)
        j_idx, k_idx = np.triu_indices(n_o, k=1)
        xj = others_arr[j_idx, 0]
        yj = others_arr[j_idx, 1]
        xk = others_arr[k_idx, 0]
        yk = others_arr[k_idx, 1]
        return np.abs(x0 * (yj - yk) + xj * (yk - y0) + xk * (y0 - yj))

    # --- Initialization ---
    
    candidates = []
    
    # Pattern 1: Two parallel rows
    for y0 in range(-15, 15):
        for gap in [1, 2, 3, 4, 5, 6, 7, 8]:
            y1 = y0 + gap
            if -30 <= y1 <= 30:
                for x_start in range(-15, 15):
                    pts = []
                    for x in range(x_start, x_start + 15):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                    for x in range(x_start, x_start + 15):
                        if -30 <= x <= 30:
                            pts.append((x, y1))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)
                        
    # Pattern 2: Three parallel rows
    for y0 in range(-15, 15):
        for gap in [1, 2, 3]:
            y1 = y0 + gap
            y2 = y0 + 2 * gap
            if -30 <= y2 <= 30:
                for x_start in range(-15, 15):
                    pts = []
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y1))
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y2))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)

    # Pattern 3: Grids
    for w in range(4, 9):
        for h in range(4, 9):
            if w * h >= 30:
                for ox in range(-15, 15):
                    for oy in range(-15, 15):
                        pts = []
                        for x in range(w):
                            for y in range(h):
                                px = ox + x
                                py = oy + y
                                if -30 <= px <= 30 and -30 <= py <= 30:
                                    pts.append((px, py))
                        if len(pts) >= 30:
                            pts = pts[:30]
                            if len(set(pts)) == 30:
                                candidates.append(pts)

    # Pattern 4: Sheared grids (affine transforms of grids)
    for w in range(4, 8):
        for h in range(4, 8):
            if w * h >= 30:
                for ox in range(-10, 10):
                    for oy in range(-10, 10):
                        for shear in [-2, -1, 0, 1, 2]:
                            pts = []
                            for x in range(w):
                                for y in range(h):
                                    px = ox + x + shear * y
                                    py = oy + y
                                    if -30 <= px <= 30 and -30 <= py <= 30:
                                        pts.append((px, py))
                            if len(pts) >= 30:
                                pts = pts[:30]
                                if len(set(pts)) == 30:
                                    candidates.append(pts)

    # Pattern 5: Clustered random points
    random.seed(42)
    for _ in range(50):
        cx = random.randint(-20, 20)
        cy = random.randint(-20, 20)
        spread = random.randint(3, 8)
        pts = []
        for _ in range(30):
            x = cx + random.randint(-spread, spread)
            y = cy + random.randint(-spread, spread)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) == 30 and len(set(pts)) == 30:
            candidates.append(pts)

    if not candidates:
        candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])

    random.seed(42)
    best_score = 0
    best_points = None
    best_counts = None
    best_area = 0
    
    # Evaluate initial candidates
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        counts, area, count = compute_all_areas(cand)
        if count > best_score:
            best_score = count
            best_points = cand[:]
            best_counts = counts
            best_area = area

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_counts, best_area, best_score = compute_all_areas(best_points)

    # --- Local Search ---
    
    current_counts = best_counts.copy()
    current_best_area = best_area
    current_best_count = best_score
    current_points = best_points[:]
    
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    point_set = set(current_points)
    
    def get_new_max_count(counts, old_areas, new_areas, current_max_count):
        """Efficiently compute new max count after a point move."""
        # Build delta counter
        delta = Counter()
        for a in old_areas:
            if a > 0:
                delta[a] -= 1
        for a in new_areas:
            if a > 0:
                delta[a] += 1
        
        if not delta:
            return current_max_count
        
        changed_areas = set(delta.keys())
        max_new_count = 0
        
        # Check changed areas
        for a in changed_areas:
            if a == 0:
                continue
            new_c = counts.get(a, 0) + delta[a]
            if new_c > max_new_count:
                max_new_count = new_c
                
        # Check unchanged areas - find max excluding changed
        max_unchanged = 0
        for a, c in counts.items():
            if a not in changed_areas:
                if c > max_unchanged:
                    max_unchanged = c
                    
        return max(max_new_count, max_unchanged)

    def local_search(points, counts, best_area, best_count, max_iterations=50):
        """Perform local search to improve the configuration."""
        pts = points[:]
        cnts = counts.copy()
        b_area = best_area
        b_count = best_count
        pset = set(pts)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
                
            improved = False
            point_order = list(range(30))
            random.shuffle(point_order)
            
            for i in point_order:
                if time.time() - start_time > DEADLINE:
                    break
                    
                old_point = pts[i]
                
                # Get others as numpy array
                others_idx = [j for j in range(30) if j != i]
                others_pts = np.array([pts[j] for j in others_idx], dtype=np.int64)
                
                # Compute old areas
                old_areas = compute_areas_for_point(old_point, others_pts)
                
                # Generate candidates - focus on areas near the target
                candidates_to_try = []
                
                # 1. Local perturbations
                for dx in range(-6, 7):
                    for dy in range(-6, 7):
                        nx = old_point[0] + dx
                        ny = old_point[1] + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            np_ = (nx, ny)
                            if np_ != old_point and np_ not in pset:
                                candidates_to_try.append(np_)
                                
                # 2. Random valid points
                for _ in range(15):
                    np_ = random.choice(all_valid_points)
                    if np_ != old_point and np_ not in pset:
                        candidates_to_try.append(np_)
                
                random.shuffle(candidates_to_try)
                
                # Try up to 25 candidates
                for new_point in candidates_to_try[:25]:
                    if time.time() - start_time > DEADLINE:
                        break
                    
                    x0, y0 = new_point
                    new_areas = compute_areas_for_point(new_point, others_pts)
                    
                    new_max_count = get_new_max_count(cnts, old_areas, new_areas, b_count)
                    
                    if new_max_count > b_count:
                        # Accept move
                        for a in old_areas:
                            if a > 0:
                                cnts[a] -= 1
                                if cnts[a] == 0:
                                    del cnts[a]
                        for a in new_areas:
                            if a > 0:
                                cnts[a] += 1
                                
                        pset.remove(old_point)
                        pset.add(new_point)
                        pts[i] = new_point
                        b_count = new_max_count
                        b_area, _ = cnts.most_common(1)[0]
                        
                        improved = True
                        break  # Move to next point
        
            if not improved:
                break
                
        return pts, cnts, b_area, b_count

    # Multi-pass refinement with progressively tighter neighborhoods
    for pass_num in range(3):
        if time.time() - start_time > DEADLINE:
            break
            
        # Determine neighborhood size based on pass
        if pass_num == 0:
            max_iter = 30
        elif pass_num == 1:
            max_iter = 20
        else:
            max_iter = 15
            
        current_points, current_counts, current_best_area, current_best_count = local_search(
            current_points, current_counts, current_best_area, current_best_count, max_iter
        )

    # Final polish: try single-point moves with wider neighborhood
    if time.time() - start_time < DEADLINE - 5:
        pts = current_points[:]
        cnts = current_counts.copy()
        b_area = current_best_area
        b_count = current_best_count
        pset = set(pts)
        
        for i in range(30):
            if time.time() - start_time > DEADLINE:
                break
                
            old_point = pts[i]
            others_idx = [j for j in range(30) if j != i]
            others_pts = np.array([pts[j] for j in others_idx], dtype=np.int64)
            old_areas = compute_areas_for_point(old_point, others_pts)
            
            # Try all points in a larger neighborhood
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if time.time() - start_time > DEADLINE:
                        break
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ != old_point and np_ not in pset:
                            new_areas = compute_areas_for_point(np_, others_pts)
                            new_max_count = get_new_max_count(cnts, old_areas, new_areas, b_count)
                            if new_max_count > b_count:
                                for a in old_areas:
                                    if a > 0:
                                        cnts[a] -= 1
                                        if cnts[a] == 0:
                                            del cnts[a]
                                for a in new_areas:
                                    if a > 0:
                                        cnts[a] += 1
                                pset.remove(old_point)
                                pset.add(np_)
                                pts[i] = np_
                                b_count = new_max_count
                                b_area, _ = cnts.most_common(1)[0]
                                break
                if time.time() - start_time > DEADLINE:
                    break

        if b_count > current_best_count:
            current_points = pts
            current_counts = cnts
            current_best_area = b_area
            current_best_count = b_count

    return current_points
