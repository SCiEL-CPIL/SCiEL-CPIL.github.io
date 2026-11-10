import numpy as np
import time
import random
from math import gcd
from collections import defaultdict

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _line_intersection(l1, l2):
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x_num = B1 * C2 - B2 * C1
    y_num = A2 * C1 - A1 * C2
    if x_num % det != 0 or y_num % det != 0:
        return None
    x = x_num // det
    y = y_num // det
    # Allow a reasonably wide range for intersection points
    if abs(x) > 300 or abs(y) > 300:
        return None
    return (x, y)

def _generate_line_pool():
    pool = set()
    # Coefficients up to 5 to cover simple slopes
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0:
                continue
            # Constant term range
            # Points are expected to be in roughly [-15, 20]
            # A*x + B*y + C = 0 => C = -(A*x + B*y)
            # Max |C| approx 5*20 + 5*20 = 200? 
            # Let's use a safe range [-100, 100]
            for C in range(-100, 101):
                canon = _normalize_line(A, B, C)
                if canon:
                    pool.add(canon)
    return list(pool)

def _make_grid(rows, cols):
    pts = []
    for x in range(cols):
        for y in range(rows):
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def _make_random(n, seed=42):
    rng = random.Random(seed)
    pts = []
    seen = set()
    while len(pts) < n:
        x = rng.randint(-10, 15)
        y = rng.randint(-10, 15)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def _run_search(initial_pts, pool, point_line_map, deadline):
    n_pts = len(initial_pts)
    n_pool = len(pool)
    
    pts = initial_pts.copy()
    # Track which point is at which index for distinctness check
    point_indices = {}
    for i in range(n_pts):
        pt = (pts[i, 0], pts[i, 1])
        point_indices[pt] = i

    # Init counts
    counts = np.zeros(n_pool, dtype=np.int64)
    for i in range(n_pts):
        x, y = pts[i]
        indices = point_line_map.get((x, y), np.array([], dtype=int))
        if len(indices) > 0:
            counts[indices] += 1
            
    def get_top_k_sum(c, k=27):
        if n_pool < k:
            return int(np.sum(c))
        top_k = np.partition(c, -k)[-k:]
        return int(np.sum(top_k))
        
    current_score = get_top_k_sum(counts)
    best_score = current_score
    best_pts = pts.copy()
    best_counts = counts.copy()
    
    start_time = time.time()
    rng = random.Random(123)
    
    T = 1.0
    T_min = 0.01
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
            
        i = rng.randint(0, n_pts - 1)
        old_x, old_y = pts[i]
        old_pt = (old_x, old_y)
        
        # Propose new point
        if rng.random() < 0.8:
            dx = rng.randint(-2, 2)
            dy = rng.randint(-2, 2)
            new_x, new_y = old_x + dx, old_y + dy
        else:
            new_x = rng.randint(-12, 18)
            new_y = rng.randint(-12, 18)
            
        new_pt = (new_x, new_y)
        
        # Check bounds for map (Map covers -15 to 20)
        if new_x < -15 or new_x > 20 or new_y < -15 or new_y > 20:
            continue
            
        # Check distinctness
        if new_pt in point_indices:
            if point_indices[new_pt] == i:
                # No change
                continue
            else:
                # Collision
                continue
        
        # Update counts
        old_indices = point_line_map.get(old_pt, np.array([], dtype=int))
        new_indices = point_line_map.get(new_pt, np.array([], dtype=int))
        
        if len(old_indices) > 0:
            counts[old_indices] -= 1
        if len(new_indices) > 0:
            counts[new_indices] += 1
            
        new_score = get_top_k_sum(counts)
        
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            # Accept
            point_indices.pop(old_pt)
            point_indices[new_pt] = i
            pts[i] = [new_x, new_y]
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_counts = counts.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            T *= 0.999
            if T < T_min:
                T = T_min
        else:
            # Revert
            if len(old_indices) > 0:
                counts[old_indices] += 1
            if len(new_indices) > 0:
                counts[new_indices] -= 1
            no_improve += 1
            
    # Extract best lines
    top_indices = np.argsort(best_counts)[-27:]
    best_lines = [pool[i] for i in top_indices]
    
    return best_pts, np.array(best_lines, dtype=np.int64), best_score

def _direct_construction_v1():
    """
    Strategy:
    1. Define a set of 54 points on a grid or specific structure.
    2. Find all lines that pass through at least 2 of these points.
    3. Select the top 27 lines (by number of points they pass through).
    """
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    pts = np.array(points, dtype=np.int64)
    
    candidate_lines = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 10 and abs(B) <= 10 and abs(C) <= 50:
                canon = _normalize_line(A, B, C)
                if canon:
                    candidate_lines.add(canon)
    
    line_scores = {}
    for line in candidate_lines:
        A, B, C = line
        cnt = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                cnt += 1
        line_scores[line] = cnt
    
    sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
    
    selected_lines = []
    seen = set()
    for line, score in sorted_lines:
        if score < 2:
            break
        if line not in seen:
            selected_lines.append(line)
            seen.add(line)
        if len(selected_lines) == 27:
            break
    
    if len(selected_lines) < 27:
        # Pad with other lines if necessary
        for line, score in sorted_lines:
            if line not in seen:
                selected_lines.append(line)
                seen.add(line)
            if len(selected_lines) == 27:
                break
    
    return pts, np.array(selected_lines, dtype=np.int64)

def _direct_construction_v2():
    """
    Alternative: Use a set of lines first, then optimize points.
    Let's try a different point structure: a triangular or hexagonal lattice might be better for 60-degree lines?
    However, integer coordinates constrain us to slopes p/q.
    Let's try a 3x18 grid.
    """
    points = []
    for x in range(3):
        for y in range(18):
            points.append((x, y))
    pts = np.array(points, dtype=np.int64)
    
    candidate_lines = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 10 and abs(B) <= 10 and abs(C) <= 50:
                canon = _normalize_line(A, B, C)
                if canon:
                    candidate_lines.add(canon)
    
    line_scores = {}
    for line in candidate_lines:
        A, B, C = line
        cnt = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                cnt += 1
        line_scores[line] = cnt
    
    sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
    
    selected_lines = []
    seen = set()
    for line, score in sorted_lines:
        if score < 2:
            break
        if line not in seen:
            selected_lines.append(line)
            seen.add(line)
        if len(selected_lines) == 27:
            break
    
    if len(selected_lines) < 27:
        for line, score in sorted_lines:
            if line not in seen:
                selected_lines.append(line)
                seen.add(line)
            if len(selected_lines) == 27:
                break
                
    return pts, np.array(selected_lines, dtype=np.int64)

def _direct_construction_v3():
    """
    Let's try a 2x27 grid.
    """
    points = []
    for x in range(2):
        for y in range(27):
            points.append((x, y))
    pts = np.array(points, dtype=np.int64)
    
    candidate_lines = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 10 and abs(B) <= 10 and abs(C) <= 50:
                canon = _normalize_line(A, B, C)
                if canon:
                    candidate_lines.add(canon)
    
    line_scores = {}
    for line in candidate_lines:
        A, B, C = line
        cnt = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                cnt += 1
        line_scores[line] = cnt
    
    sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
    
    selected_lines = []
    seen = set()
    for line, score in sorted_lines:
        if score < 2:
            break
        if line not in seen:
            selected_lines.append(line)
            seen.add(line)
        if len(selected_lines) == 27:
            break
    
    if len(selected_lines) < 27:
        for line, score in sorted_lines:
            if line not in seen:
                selected_lines.append(line)
                seen.add(line)
            if len(selected_lines) == 27:
                break
                
    return pts, np.array(selected_lines, dtype=np.int64)

def _count_incidences(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # Vectorized check
    # lines: (27, 3), points: (54, 2)
    # We want to check A*x + B*y + C == 0 for all pairs
    # This is a 27x54 matrix of values
    vals = lns[:, 0, None] * pts[:, 0, None] + lns[:, 1, None] * pts[:, 1, None] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580 # 20s margin
    
    pool = _generate_line_pool()
    n_pool = len(pool)
    
    # Precompute point_line_map for the bounding box
    # Box: x in [-15, 20], y in [-15, 20]
    A_pool = np.array([l[0] for l in pool])
    B_pool = np.array([l[1] for l in pool])
    C_pool = np.array([l[2] for l in pool])
    
    point_line_map = {}
    for x in range(-15, 21):
        for y in range(-15, 21):
            vals = A_pool * x + B_pool * y + C_pool
            mask = vals == 0
            indices = np.where(mask)[0]
            point_line_map[(x, y)] = indices
            
    configs = [
        _make_grid(9, 6), # 6x9
        _make_grid(18, 3), # 3x18
        _make_grid(27, 2), # 2x27
        _make_random(54)
    ]
    
    # Also try direct constructions as initial configs for search
    try:
        pts1, lines1 = _direct_construction_v1()
        configs.append(pts1)
    except Exception:
        pass
    try:
        pts2, lines2 = _direct_construction_v2()
        configs.append(pts2)
    except Exception:
        pass
    try:
        pts3, lines3 = _direct_construction_v3()
        configs.append(pts3)
    except Exception:
        pass
        
    best_overall_pts = None
    best_overall_lines = None
    best_overall_score = -1
    
    # Evaluate direct constructions immediately
    for i in range(3):
        try:
            if i == 0:
                pts, lines = _direct_construction_v1()
            elif i == 1:
                pts, lines = _direct_construction_v2()
            else:
                pts, lines = _direct_construction_v3()
            score = _count_incidences(pts, lines)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts
                best_overall_lines = lines
        except Exception:
            pass
            
    time_per_config = (deadline - start_time) / len(configs)
    
    for i, init_pts in enumerate(configs):
        t_end = start_time + time_per_config * (i + 1)
        if i == len(configs) - 1:
            t_end = deadline - 5
            
        try:
            pts, lines, score = _run_search(init_pts, pool, point_line_map, t_end)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts
                best_overall_lines = lines
        except Exception:
            pass
            
    if best_overall_pts is None:
        # Fallback
        best_overall_pts = _make_grid(9, 6)
        # Calculate lines for fallback
        counts = np.zeros(n_pool, dtype=np.int64)
        for i in range(54):
            x, y = best_overall_pts[i]
            indices = point_line_map.get((x, y), np.array([], dtype=int))
            if len(indices) > 0:
                counts[indices] += 1
        top_indices = np.argsort(counts)[-27:]
        best_overall_lines = np.array([pool[i] for i in top_indices], dtype=np.int64)
        
    return best_overall_pts, best_overall_lines
