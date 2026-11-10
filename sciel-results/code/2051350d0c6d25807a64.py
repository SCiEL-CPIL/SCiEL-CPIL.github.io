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
    vals = lns[:, 0, None, None] * pts[:, 0, None] + lns[:, 1, None, None] * pts[:, 1, None] + lns[:, 2, None, None]
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

def _greedy_line_selection(pool, n_lines=27, deadline=None):
    """
    Greedily select n_lines lines from pool to maximize sum of pairwise intersection multiplicities.
    Score of a candidate line given current selection = sum over existing lines of (1 if they intersect at an integer point within bounds).
    This approximates the number of unique intersection points created.
    """
    selected = []
    selected_set = set()
    
    # Precompute pairwise intersection points for all pairs in pool? Too big.
    # Instead, compute on the fly.
    
    # First line: pick the one that has the most potential intersections with others?
    # Or just pick randomly from top candidates.
    # Let's compute for each line in pool, how many other lines in pool it intersects with at an integer point.
    # This is O(N^2) where N is pool size. If pool is ~1000, that's 1M ops, fine.
    
    n_pool = len(pool)
    if n_pool < n_lines:
        return list(pool)
    
    # Compute potential score for each line: number of other lines it intersects with
    potential = [0] * n_pool
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            pt = _line_intersection(pool[i], pool[j])
            if pt is not None:
                potential[i] += 1
                potential[j] += 1
    
    # Sort indices by potential descending
    order = sorted(range(n_pool), key=lambda i: -potential[i])
    
    # Greedy selection
    for idx in order:
        if len(selected) >= n_lines:
            break
        if idx in selected_set:
            continue
        selected.append(pool[idx])
        selected_set.add(idx)
        
        if deadline and time.time() > deadline:
            break
    
    # If we didn't fill up, add remaining
    if len(selected) < n_lines:
        for idx in order:
            if len(selected) >= n_lines:
                break
            if idx not in selected_set:
                selected.append(pool[idx])
                selected_set.add(idx)
    
    return selected[:n_lines]

def _place_points_greedy(lines, n_points=54):
    """
    Given a set of lines, find the n_points lattice points that maximize incidences.
    Each point's value = number of lines it lies on.
    Greedily pick the point with highest value, then remove it from consideration (but keep lines).
    Actually, we just pick the top n_points distinct lattice points by their line-coverage count.
    """
    # Find all integer intersection points of pairs of lines
    # Also consider points on single lines? No, a point on 1 line is less valuable than on 2.
    # We want points that lie on many lines.
    
    point_count = defaultdict(int)
    
    n_lines = len(lines)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_count[pt] += 1
    
    # Also, what if a line has many points? We only care about intersections for high-value points.
    # But we need 54 points. If we have fewer than 54 intersection points, we need to add points on single lines.
    
    # Sort points by count descending
    sorted_points = sorted(point_count.items(), key=lambda x: -x[1])
    
    points = []
    seen = set()
    
    # First, add all intersection points
    for pt, cnt in sorted_points:
        if len(points) >= n_points:
            break
        if pt not in seen:
            points.append(pt)
            seen.add(pt)
    
    # If we need more points, find points on individual lines
    if len(points) < n_points:
        # For each line, find integer points on it that are not already selected
        # Line Ax + By + C = 0. If gcd(A,B) divides C, there are integer solutions.
        # We can parameterize.
        for line in lines:
            A, B, C = line
            g = gcd(abs(A), abs(B))
            if g == 0 or C % g != 0:
                continue
            # A/g * x + B/g * y = -C/g
            a = A // g
            b = B // g
            c = -C // g
            # Find one solution
            # Use extended euclid to find x0, y0 such that a*x + b*y = 1
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g2, x1, y1 = egcd(a, b)
            # a*x1 + b*y1 = g2 = 1 (since gcd(a,b)=1)
            x0 = x1 * c
            y0 = y1 * c
            # General solution: x = x0 + b*t, y = y0 - a*t
            # Try t in a range
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
    Local search: swap a point to a nearby intersection, or swap a line to a nearby line.
    """
    rng = random.Random(42)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence matrix
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
    
    total_inc = int(inc.sum())
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Candidate points: intersections of current lines + nearby
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                # Add neighbors
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    
    # Candidate lines: nearby lines
    cand_lines = set()
    for A in range(-10, 11):
        for B in range(-10, 11):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 31):
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.add(canon)
    
    cand_points = list(cand_points)
    cand_lines = list(cand_lines)
    
    T_start = 3.0
    T_end = 0.1
    iter_count = 0
    n_iters = 100000
    
    while time.time() < deadline - 10:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        if rng.random() < 0.5:
            # Point move
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            
            if new_pos == old_pos:
                continue
            # Check duplicate
            if any(tuple(pts[k]) == new_pos for k in range(n_pts) if k != i):
                continue
            
            delta = 0
            for j in range(n_lines):
                was = inc[i, j]
                A, B, C = lns[j]
                now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
            
            new_total = total_inc + delta
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                total_inc = new_total
                pts[i] = new_pos
                for j in range(n_lines):
                    A, B, C = lns[j]
                    inc[i, j] = (A * new_pos[0] + B * new_pos[1] + C == 0)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
        else:
            # Line move
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines[rng.randint(0, len(cand_lines) - 1)]
            new_canon = _normalize_line(int(new_line[0]), int(new_line[1]), int(new_line[2]))
            
            if new_canon is None or new_canon == old_canon:
                continue
            # Check duplicate
            if any(_normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_canon for k in range(n_lines) if k != j):
                continue
            
            delta = 0
            A, B, C = new_line
            for i in range(n_pts):
                was = inc[i, j]
                now = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
            
            new_total = total_inc + delta
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                total_inc = new_total
                lns[j] = new_line
                for i in range(n_pts):
                    inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
    
    return best_pts, best_lns, best_total

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # Build a pool of lines with small coefficients
    pool = []
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                canon = _normalize_line(A, B, C)
                if canon:
                    pool.append(canon)
    
    # Deduplicate
    seen = set()
    unique_pool = []
    for line in pool:
        if line not in seen:
            seen.add(line)
            unique_pool.append(line)
    pool = unique_pool
    
    # Greedy line selection
    selected_lines = _greedy_line_selection(pool, 27, deadline - 120)
    
    # Place points greedily
    points = _place_points_greedy(selected_lines, 54)
    
    lines = np.array(selected_lines, dtype=np.int64)
    pts = np.array(points, dtype=np.int64)
    
    # Local search
    if time.time() < deadline - 30:
        ls_pts, ls_lns, ls_count = _local_search(pts, lines, start_time, deadline - 15)
        pts = ls_pts
        lines = ls_lns
    
    # Final validation
    pts = np.array(pts, dtype=np.int64)
    lines = np.array(lines, dtype=np.int64)
    
    if not _is_valid(pts, lines):
        # Fallback: simple grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
    
    return pts, lines
