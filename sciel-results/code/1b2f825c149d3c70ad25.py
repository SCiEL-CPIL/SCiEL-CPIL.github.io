import numpy as np
import time
from math import gcd
from itertools import combinations
import random
import sys

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

def _compute_incidence_matrix(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return (vals == 0).astype(np.int64)

def _incidence_count(points, lines):
    return int(np.sum(_compute_incidence_matrix(points, lines)))

def _initial_grid_diagonal():
    w, h = 6, 9
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
    
    pts = np.array(points, dtype=np.int64)
    
    lines = []
    for x in range(w):
        lines.append((1, 0, -x))
    for y in range(h):
        lines.append((0, 1, -y))
    
    diag_cands = []
    for c in range(-h+1, w):
        diag_cands.append((1, -1, -c))
    for c in range(0, w+h-1):
        diag_cands.append((1, 1, -c))
    
    scored = []
    for l in diag_cands:
        A, B, C = l
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        cnt = int(np.sum(vals == 0))
        scored.append((cnt, l))
    
    scored.sort(key=lambda x: -x[0])
    
    current_lines = set(lines)
    for cnt, l in scored:
        canon = _normalize_line(*l)
        if canon not in current_lines:
            lines.append(canon)
            current_lines.add(canon)
            if len(lines) == 27:
                break
    
    return np.array(points, dtype=np.int64), np.array(lines, dtype=np.int64)

def _local_search(points, lines, deadline):
    rng = random.Random(123)
    
    pts = points.copy()
    lns = lines.copy()
    n_pts, n_lines = 54, 27
    
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    point_set = set(map(tuple, pts.tolist()))
    line_set = set(_normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) for k in range(n_lines))
    
    # Generate candidate lines
    cand_lines = set()
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0: continue
            for C in range(-20, 21):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
                
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
            
    cand_lines_list = list(cand_lines)
    
    def get_cand_points():
        cands = set()
        for p in pts:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    cands.add((int(p[0])+dx, int(p[1])+dy))
        return list(cands)
    
    cand_pts = get_cand_points()
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    A_arr = lns[:, 0]
    B_arr = lns[:, 1]
    C_arr = lns[:, 2]
    
    while time.time() < deadline - 5:
        if rng.random() < 0.5:
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_pts[rng.randint(0, len(cand_pts)-1)]
            
            if new_pos == old_pos:
                continue
            if new_pos in point_set:
                continue
            
            new_vals = A_arr * new_pos[0] + B_arr * new_pos[1] + C_arr
            new_inc = (new_vals == 0).astype(np.int64)
            old_inc = inc_matrix[:, i]
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                point_set.discard(old_pos)
                point_set.add(new_pos)
                pts[i] = new_pos
                inc_matrix[:, i] = new_inc
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                
                if rng.random() < 0.05:
                    cand_pts = get_cand_points()
        else:
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines_list[rng.randint(0, len(cand_lines_list)-1)]
            
            if new_line == old_canon:
                continue
            if new_line in line_set:
                continue
            
            A, B, C = new_line
            new_vals = A * pts[:, 0] + B * pts[:, 1] + C
            new_inc = (new_vals == 0).astype(np.int64)
            old_inc = inc_matrix[j, :]
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                line_set.discard(old_canon)
                line_set.add(new_line)
                lns[j] = new_line
                inc_matrix[j, :] = new_inc
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                
                A_arr = lns[:, 0]
                B_arr = lns[:, 1]
                C_arr = lns[:, 2]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_lns

def _refine_points(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    point_set = set(map(tuple, pts.tolist()))
    A_arr = lns[:, 0]
    B_arr = lns[:, 1]
    C_arr = lns[:, 2]
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_pts):
            best_pos = tuple(pts[i])
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[:, i]))
            
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    new_pos = (int(pts[i, 0]) + dx, int(pts[i, 1]) + dy)
                    if new_pos == best_pos:
                        continue
                    if new_pos in point_set:
                        continue
                    
                    new_vals = A_arr * new_pos[0] + B_arr * new_pos[1] + C_arr
                    new_inc = (new_vals == 0).astype(np.int64)
                    new_sum = int(np.sum(new_inc))
                    delta = new_sum - old_sum
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = new_pos
                        best_new_inc = new_inc
            
            if best_delta > 0:
                point_set.discard(tuple(pts[i]))
                point_set.add(best_pos)
                pts[i] = best_pos
                inc_matrix[:, i] = best_new_inc
                current_score += best_delta
                improved = True
    
    return pts

def _greedy_select_lines(pts, deadline):
    """After points are fixed, greedily select the 27 best non-duplicate lines."""
    n_pts = 54
    pts_x = pts[:, 0]
    pts_y = pts[:, 1]
    
    # Generate a large pool of candidate lines
    cand_lines = set()
    for A in range(-10, 11):
        for B in range(-10, 11):
            if A == 0 and B == 0: continue
            for C in range(-30, 31):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
    
    # Add secant lines through pairs of points
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
    
    # Score all candidates
    scored = []
    for (A, B, C) in cand_lines:
        vals = A * pts_x + B * pts_y + C
        cnt = int(np.sum(vals == 0))
        if cnt > 0:
            scored.append((cnt, (A, B, C)))
    
    scored.sort(key=lambda x: -x[0])
    
    # Greedily pick top 27 non-duplicate lines
    selected = []
    seen = set()
    for cnt, line in scored:
        if line in seen:
            continue
        seen.add(line)
        selected.append(line)
        if len(selected) == 27:
            break
    
    # If we don't have 27, fill with any valid lines
    if len(selected) < 27:
        for (A, B, C) in cand_lines:
            if (A, B, C) not in seen:
                seen.add((A, B, C))
                selected.append((A, B, C))
                if len(selected) == 27:
                    break
    
    return np.array(selected, dtype=np.int64)

def _refine_lines(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    line_set = set(_normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) for k in range(n_lines))
    improved = True
    
    # Pre-generate candidate lines
    cand_lines = set()
    for A in range(-10, 11):
        for B in range(-10, 11):
            if A == 0 and B == 0: continue
            for C in range(-30, 31):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
            
    cand_lines_list = list(cand_lines)
    
    # Vectorized scoring: precompute for each candidate line the incidence with all points
    # Score matrix: cand_lines x n_pts
    cl_arr = np.array(cand_lines_list, dtype=np.int64)
    pts_x = pts[:, 0]
    pts_y = pts[:, 1]
    
    # For each candidate line, compute incidence count with all 54 points
    # vals[c, p] = A_c * x_p + B_c * y_p + C_c
    vals = cl_arr[:, 0, None] * pts_x[None, :] + cl_arr[:, 1, None] * pts_y[None, :] + cl_arr[:, 2, None]
    inc_counts = (vals == 0).sum(axis=1).astype(np.int64)
    
    while improved and time.time() < deadline - 10:
        improved = False
        for j in range(n_lines):
            best_line = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[j, :]))
            
            for idx, new_line in enumerate(cand_lines_list):
                if new_line == best_line:
                    continue
                if new_line in line_set:
                    continue
                
                new_sum = int(inc_counts[idx])
                delta = new_sum - old_sum
                
                if delta > best_delta:
                    best_delta = delta
                    best_line = new_line
                    A, B, C = new_line
                    new_vals = A * pts_x + B * pts_y + C
                    best_new_inc = (new_vals == 0).astype(np.int64)
            
            if best_delta > 0:
                old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
                line_set.discard(old_canon)
                line_set.add(best_line)
                lns[j] = best_line
                inc_matrix[j, :] = best_new_inc
                current_score += best_delta
                improved = True
    
    return lns

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    best_pts = None
    best_lns = None
    best_score = -1
    
    for restart in range(3):
        if time.time() > deadline - 60:
            break
        
        pts, lns = _initial_grid_diagonal()
        
        if restart > 0:
            rng = random.Random(restart * 1000)
            for i in range(rng.randint(3, 8)):
                idx = rng.randint(0, 53)
                pts[idx] = [rng.randint(-5, 10), rng.randint(-5, 10)]
            seen = set()
            for i in range(54):
                t = tuple(pts[i])
                if t in seen:
                    pts[i] = [i % 6, i // 6]
                seen.add(tuple(pts[i]))
        
        if time.time() < deadline - 80:
            pts = _refine_points(pts, lns, deadline - 80)
        
        if time.time() < deadline - 80:
            lns = _refine_lines(pts, lns, deadline - 80)
        
        if time.time() < deadline - 40:
            pts, lns = _local_search(pts, lns, deadline - 40)
        
        if time.time() < deadline - 10:
            pts = _refine_points(pts, lns, deadline - 10)
            # Use greedy line selection for the final pass
            lns = _greedy_select_lines(pts, deadline - 5)
        
        score = _incidence_count(pts, lns)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            best_lns = lns.copy()
    
    pts = np.array(best_pts, dtype=np.int64)
    lns = np.array(best_lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for c in range(0, 12):
            lines.append((1, 1, -c))
        lns = np.array(lines, dtype=np.int64)
    
    return pts, lns
