import numpy as np
import time
import random
from math import gcd
from collections import defaultdict
from itertools import combinations

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

def _is_valid(points, lines):
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in [p, l]:
        if not np.all(np.isfinite(arr)):
            return False
        if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
            return False
    if len(set(map(tuple, np.round(p, 6).astype(int).tolist()))) != 54:
        return False
    seen_lines = set()
    for row in np.round(l, 6).astype(int).tolist():
        canon = _normalize_line(row[0], row[1], row[2])
        if canon is None:
            return False
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
    return True

def _incidence_count(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def _line_intersection(l1, l2):
    """Return integer point (x,y) if two lines intersect at an integer point, else None."""
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
    if abs(x) > 100 or abs(y) > 100:
        return None
    return (x, y)

def _generate_line_pool(max_coeff=10, max_c=30):
    """Generate a diverse pool of lines with small integer coefficients."""
    pool = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_c, max_c + 1):
                canon = _normalize_line(A, B, C)
                if canon:
                    pool.add(canon)
    return list(pool)

def _greedy_line_selection(pool, n_lines=27, deadline=None):
    """
    Greedily select n_lines lines from pool to maximize their pairwise intersection potential.
    This aims to create a dense set of intersection points.
    """
    selected = []
    selected_set = set()
    
    n_pool = len(pool)
    if n_pool < n_lines:
        return list(pool)
    
    # Compute potential score for each line: number of other lines it intersects with at an integer point.
    # To save time, we can limit the intersection check range or use a smaller pool initially.
    # However, for ~1000 lines, O(N^2) is 1M ops, which is fast.
    potential = [0] * n_pool
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            pt = _line_intersection(pool[i], pool[j])
            if pt is not None:
                potential[i] += 1
                potential[j] += 1
    
    # Sort indices by potential descending
    order = sorted(range(n_pool), key=lambda i: -potential[i])
    
    # Greedy selection: pick lines that add the most new high-multiplicity intersection points?
    # Simple heuristic: pick lines with high potential, then refine.
    # A better heuristic: pick a line, then pick the next line that maximizes intersections with the first, etc.
    
    # Let's try a smarter greedy:
    # Start with the line with highest potential.
    # Then repeatedly add the line that maximizes the number of *new* intersection points created with the existing set.
    
    selected_idx = []
    if order:
        selected_idx.append(order[0])
    
    while len(selected_idx) < n_lines:
        best_idx = -1
        best_score = -1
        for idx in order:
            if idx in selected_idx:
                continue
            # Score: number of intersections with currently selected lines
            score = 0
            for s_idx in selected_idx:
                pt = _line_intersection(pool[idx], pool[s_idx])
                if pt is not None:
                    score += 1
            if score > best_score:
                best_score = score
                best_idx = idx
            if deadline and time.time() > deadline:
                break
        if best_idx == -1:
            # Pick next from order
            for idx in order:
                if idx not in selected_idx:
                    best_idx = idx
                    break
        if best_idx == -1:
            break
        selected_idx.append(best_idx)
        if deadline and time.time() > deadline:
            break
    
    return [pool[i] for i in selected_idx]

def _place_points_greedy(lines, n_points=54, deadline=None):
    """
    Given a set of lines, find the n_points lattice points that maximize incidences.
    1. Find all integer intersection points of pairs of lines.
    2. Count multiplicity (how many lines pass through each point).
    3. Select top n_points by multiplicity.
    4. If not enough, add points on lines.
    """
    point_count = defaultdict(int)
    
    n_lines = len(lines)
    # Check all pairs
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_count[pt] += 1
                
    # Also, points on single lines might be needed if we don't have enough intersections.
    # But intersections are generally higher value (>= 2).
    # Let's also consider points that lie on 1 line? No, we want max incidences.
    # If we have < 54 intersection points, we must add more.
    
    sorted_points = sorted(point_count.items(), key=lambda x: -x[1])
    
    points = []
    seen = set()
    
    # Add top intersection points
    for pt, cnt in sorted_points:
        if len(points) >= n_points:
            break
        if pt not in seen:
            points.append(pt)
            seen.add(pt)
    
    # If we need more points, find points on individual lines that are not already selected.
    # We prioritize points that lie on more lines, but we've already exhausted those (intersections).
    # So we just need any new point. To maximize future potential, maybe pick points on lines with high density?
    # For now, just pick arbitrary valid points on lines.
    
    if len(points) < n_points:
        for line in lines:
            A, B, C = line
            g = gcd(abs(A), abs(B))
            if g == 0 or C % g != 0:
                continue
            a = A // g
            b = B // g
            c = -C // g
            # Find one solution to a*x + b*y = c
            # Extended Euclid
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g2, x1, y1 = egcd(a, b)
            if g2 != 1: # Should not happen if we divided by gcd
                continue
            x0 = x1 * c
            y0 = y1 * c
            # General solution: x = x0 + b*t, y = y0 - a*t
            for t in range(-50, 51):
                x = x0 + b * t
                y = y0 - a * t
                pt = (x, y)
                if pt not in seen and abs(x) <= 100 and abs(y) <= 100:
                    points.append(pt)
                    seen.add(pt)
                    if len(points) >= n_points:
                        break
            if len(points) >= n_points:
                break
    
    # If still not enough, add arbitrary lattice points
    if len(points) < n_points:
        for x in range(-50, 51):
            for y in range(-50, 51):
                if len(points) >= n_points:
                    break
                pt = (x, y)
                if pt not in seen:
                    points.append(pt)
                    seen.add(pt)
    
    return points[:n_points]

def _local_search(points, lines, start_time, deadline):
    """
    Local search to improve the incidence count.
    Moves:
    1. Swap a point with a candidate point (nearby or from intersection pool).
    2. Swap a line with a candidate line (from pool).
    """
    rng = random.Random(42)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Precompute incidence matrix? 54x27 is small.
    # But we need to update it efficiently.
    # Let's just recompute the delta for a move.
    
    # Candidate lines: generated pool
    pool = _generate_line_pool(max_coeff=8, max_c=25)
    
    # Candidate points: 
    # 1. Nearby current points
    # 2. Intersections of current lines (already there)
    # 3. Intersections of current lines with candidate lines?
    # Let's stick to a local neighborhood for points to keep it fast.
    
    def get_point_candidates(pts):
        cands = set()
        for p in pts:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    cands.add((int(p[0])+dx, int(p[1])+dy))
        return list(cands)
    
    def compute_inc(pts, lns):
        vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
        return int(np.sum(vals == 0))
    
    current_score = compute_inc(pts, lns)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    T_start = 10.0
    T_end = 0.1
    iter_count = 0
    n_iters = 500000 # Adjust based on time
    
    while time.time() < deadline - 5:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        if rng.random() < 0.5:
            # Move a point
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            
            cands = get_point_candidates(pts)
            new_pos = cands[rng.randint(0, len(cands)-1)]
            
            if new_pos == old_pos:
                continue
            
            # Check duplicates
            dup = False
            for k in range(n_pts):
                if k != i and tuple(pts[k]) == new_pos:
                    dup = True
                    break
            if dup:
                continue
            
            # Calculate delta
            delta = 0
            for j in range(n_lines):
                A, B, C = lns[j]
                was = (A * old_pos[0] + B * old_pos[1] + C == 0)
                now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
                
            new_score = current_score + delta
            if new_score >= current_score or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_score = new_score
                pts[i] = new_pos
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
        else:
            # Move a line
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = pool[rng.randint(0, len(pool)-1)]
            
            if new_line == old_canon:
                continue
            
            # Check duplicates
            dup = False
            for k in range(n_lines):
                if k != j and _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_line:
                    dup = True
                    break
            if dup:
                continue
            
            # Calculate delta
            delta = 0
            A, B, C = new_line
            for i in range(n_pts):
                was = (lns[j, 0] * pts[i, 0] + lns[j, 1] * pts[i, 1] + lns[j, 2] == 0)
                now = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
                
            new_score = current_score + delta
            if new_score >= current_score or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_score = new_score
                lns[j] = new_line
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    
    return best_pts, best_lns

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Generate Line Pool
    pool = _generate_line_pool(max_coeff=10, max_c=30)
    
    # 2. Greedy Line Selection
    # We want lines that intersect frequently.
    selected_lines = _greedy_line_selection(pool, n_lines=27, deadline=deadline - 300)
    
    # 3. Place Points Greedily
    points = _place_points_greedy(selected_lines, n_points=54, deadline=deadline - 200)
    
    pts = np.array(points, dtype=np.int64)
    lns = np.array(selected_lines, dtype=np.int64)
    
    # 4. Local Search
    if time.time() < deadline - 10:
        pts, lns = _local_search(pts, lns, start_time, deadline - 5)
        
    # 5. Final Validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback: simple grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
        return pts, lines
        
    return pts, lns
