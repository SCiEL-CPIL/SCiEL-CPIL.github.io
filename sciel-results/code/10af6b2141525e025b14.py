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
    if abs(x) > 100 or abs(y) > 100:
        return None
    return (x, y)

def _build_line_pool(max_coeff=10, max_const=50):
    pool = []
    seen = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_const, max_const + 1):
                canon = _normalize_line(A, B, C)
                if canon and canon not in seen:
                    seen.add(canon)
                    pool.append(canon)
    return pool

def _evaluate(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def _optimize_points_given_lines(lines, n_points=54):
    lns = np.asarray(lines, dtype=np.int64)
    n_lines = len(lns)
    point_line_count = defaultdict(int)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt is not None:
                point_line_count[pt] += 1
    sorted_points = sorted(point_line_count.items(), key=lambda x: -x[1])
    points = []
    seen = set()
    for pt, cnt in sorted_points:
        if len(points) >= n_points:
            break
        if pt not in seen:
            points.append(pt)
            seen.add(pt)
    if len(points) < n_points:
        for line in lines:
            A, B, C = line
            g = gcd(abs(A), abs(B))
            if g == 0 or C % g != 0:
                continue
            a = A // g
            b = B // g
            c = -C // g
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            g2, x1, y1 = egcd(a, b)
            x0 = x1 * c
            y0 = y1 * c
            for t in range(-100, 101):
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
    if len(points) < n_points:
        for x in range(-100, 101):
            for y in range(-100, 101):
                if len(points) >= n_points:
                    break
                pt = (x, y)
                if pt not in seen:
                    points.append(pt)
                    seen.add(pt)
    return points[:n_points]

def _select_lines_for_points(points, line_pool, n_lines=27):
    pts = np.asarray(points, dtype=np.int64)
    n_pts = len(pts)
    line_scores = []
    for idx, line in enumerate(line_pool):
        A, B, C = line
        count = 0
        for i in range(n_pts):
            if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                count += 1
        line_scores.append((count, idx))
    line_scores.sort(key=lambda x: -x[0])
    selected = []
    selected_set = set()
    for count, idx in line_scores:
        if len(selected) >= n_lines:
            break
        if idx in selected_set:
            continue
        selected.append(line_pool[idx])
        selected_set.add(idx)
    if len(selected) < n_lines:
        for count, idx in line_scores:
            if len(selected) >= n_lines:
                break
            if idx not in selected_set:
                selected.append(line_pool[idx])
                selected_set.add(idx)
    return selected[:n_lines]

def _alternating_optimization(line_pool, n_points=54, n_lines=27, deadline=None, max_iter=50):
    rng = random.Random(123)
    # Start with a grid-like structure for diversity
    init_pts = [(i, j) for i in range(6) for j in range(9)]
    init_lines = [(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)]
    # Add some diagonals to reach 27
    for i in range(12):
        init_lines.append((1, 1, -i))
    init_lines = init_lines[:27]
    
    points = _optimize_points_given_lines(init_lines, n_points)
    best_score = _evaluate(points, init_lines)
    best_points = [p for p in points]
    best_lines = [l for l in init_lines]
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        new_lines = _select_lines_for_points(points, line_pool, n_lines)
        new_points = _optimize_points_given_lines(new_lines, n_points)
        score = _evaluate(new_points, new_lines)
        if score > best_score:
            best_score = score
            best_points = [p for p in new_points]
            best_lines = [l for l in new_lines]
            points = new_points
            lines = new_lines
        else:
            # Perturb
            if rng.random() < 0.5:
                i = rng.randint(0, len(points) - 1)
                # Move to random nearby point
                dx, dy = rng.randint(-3, 3), rng.randint(-3, 3)
                new_pt = (points[i][0] + dx, points[i][1] + dy)
                if not any(p == new_pt for k, p in enumerate(points) if k != i):
                    points[i] = new_pt
                    score = _evaluate(points, lines)
                    if score > best_score:
                        best_score = score
                        best_points = [p for p in points]
                        best_lines = [l for l in lines]
            else:
                j = rng.randint(0, n_lines - 1)
                cand_line = line_pool[rng.randint(0, len(line_pool) - 1)]
                if _normalize_line(*cand_line) != _normalize_line(*lines[j]):
                    if not any(_normalize_line(*l) == _normalize_line(*cand_line) for k, l in enumerate(lines) if k != j):
                        lines[j] = cand_line
                        score = _evaluate(points, lines)
                        if score > best_score:
                            best_score = score
                            best_points = [p for p in points]
                            best_lines = [l for l in lines]
    return best_points, best_lines, best_score

def _local_search(points, lines, start_time, deadline):
    rng = random.Random(42)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    total_inc = _evaluate(pts, lns)
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Build candidate pools
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    cand_points = list(cand_points)
    
    cand_lines = set()
    for A in range(-12, 13):
        for B in range(-12, 13):
            if A == 0 and B == 0: continue
            for C in range(-40, 41):
                canon = _normalize_line(A, B, C)
                if canon: cand_lines.add(canon)
    cand_lines = list(cand_lines)
    
    T_start = 5.0
    T_end = 0.1
    iter_count = 0
    n_iters = 200000
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
            
    while time.time() < deadline - 10:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        if rng.random() < 0.5:
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            if new_pos == old_pos: continue
            if any(tuple(pts[k]) == new_pos for k in range(n_pts) if k != i): continue
            
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
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines[rng.randint(0, len(cand_lines) - 1)]
            new_canon = _normalize_line(int(new_line[0]), int(new_line[1]), int(new_line[2]))
            if new_canon is None or new_canon == old_canon: continue
            if any(_normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_canon for k in range(n_lines) if k != j): continue
            
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
    line_pool = _build_line_pool(max_coeff=10, max_const=50)
    
    # Phase 1: Alternating Optimization
    if time.time() < deadline - 180:
        alt_points, alt_lines, alt_score = _alternating_optimization(
            line_pool, n_points=54, n_lines=27, deadline=deadline - 180, max_iter=100
        )
        pts = np.array(alt_points, dtype=np.int64)
        lines = np.array(alt_lines, dtype=np.int64)
    else:
        # Fallback
        init_pts = [(i, j) for i in range(6) for j in range(9)]
        init_lines = [(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -i) for i in range(12)]
        pts = np.array(init_pts, dtype=np.int64)
        lines = np.array(init_lines, dtype=np.int64)
    
    # Phase 2: Local Search Polish
    if time.time() < deadline - 30:
        ls_pts, ls_lns, ls_count = _local_search(pts, lines, start_time, deadline - 10)
        pts = ls_pts
        lines = ls_lns
    
    pts = np.array(pts, dtype=np.int64)
    lines = np.array(lines, dtype=np.int64)
    
    if not _is_valid(pts, lines):
        # Fallback
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
    
    return pts, lines
