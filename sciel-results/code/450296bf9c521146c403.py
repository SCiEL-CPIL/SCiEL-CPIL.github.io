import numpy as np
import time
from math import gcd
from itertools import combinations
import random
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

def _compute_incidence_matrix(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return (vals == 0).astype(np.int64)

def _incidence_count(points, lines):
    return int(np.sum(_compute_incidence_matrix(points, lines)))

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

def _secant_lines_from_points(pts, max_coeff=12, max_const=60):
    """Generate all lines through pairs of points, filtered by coefficient size."""
    lines = set()
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(pts[i, 0]), int(pts[i, 1])
            x2, y2 = int(pts[j, 0]), int(pts[j, 1])
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= max_coeff and abs(B) <= max_coeff and abs(C) <= max_const:
                canon = _normalize_line(A, B, C)
                if canon:
                    lines.add(canon)
    return list(lines)

def _grid_lines(w, h):
    """Generate axis-aligned and diagonal lines for a w x h grid."""
    lines = set()
    for x in range(w):
        lines.add((1, 0, -x))
    for y in range(h):
        lines.add((0, 1, -y))
    # Diagonals x+y=c
    for c in range(0, w + h - 1):
        lines.add((1, 1, -c))
    # Diagonals x-y=c
    for c in range(-h + 1, w):
        lines.add((1, -1, -c))
    return list(lines)

def _score_lines_for_points(pts, candidate_lines):
    """Score each candidate line by number of points it passes through."""
    scores = {}
    for line in candidate_lines:
        A, B, C = line
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        scores[line] = int(np.sum(vals == 0))
    return scores

def _greedy_select_lines(pts, n_lines=27, extra_candidates=None):
    """Select top n_lines lines by incidence count, no duplicates."""
    n_pts = len(pts)
    cand_lines = set()
    
    # Secant lines from point pairs (primary source)
    secants = _secant_lines_from_points(pts, max_coeff=10, max_const=50)
    cand_lines.update(secants)
    
    # Systematic lines
    for A in range(-6, 7):
        for B in range(-6, 7):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                c = _normalize_line(A, B, C)
                if c:
                    cand_lines.add(c)
    
    # Extra candidates if provided
    if extra_candidates:
        cand_lines.update(extra_candidates)
    
    cand_lines_list = list(cand_lines)
    
    # Score each line
    scored = []
    for line in cand_lines_list:
        A, B, C = line
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        cnt = int(np.sum(vals == 0))
        scored.append((cnt, line))
    
    scored.sort(key=lambda x: -x[0])
    
    # Greedy selection
    selected = []
    seen = set()
    for cnt, line in scored:
        if line not in seen:
            selected.append(line)
            seen.add(line)
            if len(selected) == n_lines:
                break
    
    # Pad if needed
    if len(selected) < n_lines:
        for line in cand_lines_list:
            if line not in seen:
                selected.append(line)
                seen.add(line)
                if len(selected) == n_lines:
                    break
    
    return np.array(selected[:n_lines], dtype=np.int64)

def _initial_grid_diagonal():
    w, h = 6, 9
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
    
    pts = np.array(points, dtype=np.int64)
    
    # Use grid lines + secants
    grid_lns = _grid_lines(w, h)
    secants = _secant_lines_from_points(pts, max_coeff=5, max_const=20)
    
    all_cands = grid_lns + secants
    scores = _score_lines_for_points(pts, all_cands)
    
    # Sort by score
    sorted_lines = sorted(scores.items(), key=lambda x: -x[1])
    
    lines = []
    seen = set()
    for line, score in sorted_lines:
        if score < 2:
            break
        canon = _normalize_line(*line)
        if canon and canon not in seen:
            lines.append(canon)
            seen.add(canon)
            if len(lines) == 27:
                break
    
    # Pad if needed
    if len(lines) < 27:
        for line in all_cands:
            canon = _normalize_line(*line)
            if canon and canon not in seen:
                lines.append(canon)
                seen.add(canon)
                if len(lines) == 27:
                    break
    
    return pts, np.array(lines[:27], dtype=np.int64)

def _get_intersection_points(lines, max_coord=50):
    """Get all integer intersection points of the given lines."""
    pts = set()
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                x, y = pt
                if abs(x) <= max_coord and abs(y) <= max_coord:
                    pts.add(pt)
    return pts

def _local_search(points, lines, deadline):
    rng = random.Random(42)
    
    pts = points.copy()
    lns = lines.copy()
    n_pts, n_lines = 54, 27
    
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Build candidate line pool: secants + systematic
    secants = _secant_lines_from_points(pts, max_coeff=8, max_const=40)
    sys_lines = set()
    for A in range(-6, 7):
        for B in range(-6, 7):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                c = _normalize_line(A, B, C)
                if c:
                    sys_lines.add(c)
    
    cand_lines = set(secants) | sys_lines
    cand_lines_list = list(cand_lines)
    
    # Precompute arrays
    cand_A = np.array([l[0] for l in cand_lines_list], dtype=np.int64)
    cand_B = np.array([l[1] for l in cand_lines_list], dtype=np.int64)
    cand_C = np.array([l[2] for l in cand_lines_list], dtype=np.int64)
    
    # Get intersection points as preferred locations
    inter_pts = _get_intersection_points(lns.tolist(), max_coord=30)
    inter_pts_list = list(inter_pts)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9997
    
    last_greedy_check = time.time()
    greedy_interval = 3.0
    
    while time.time() < deadline - 10:
        if rng.random() < 0.65:
            # Move point
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            
            # Prefer intersection points, then nearby
            if inter_pts_list and rng.random() < 0.4:
                new_pos = inter_pts_list[rng.randint(0, len(inter_pts_list) - 1)]
            else:
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                new_pos = (int(pts[i, 0]) + dx, int(pts[i, 1]) + dy)
            
            if new_pos == old_pos:
                continue
            
            dup = False
            for k in range(n_pts):
                if k != i and tuple(pts[k]) == new_pos:
                    dup = True
                    break
            if dup:
                continue
            
            A = lns[:, 0]
            B = lns[:, 1]
            C = lns[:, 2]
            new_vals = A * new_pos[0] + B * new_pos[1] + C
            new_inc = (new_vals == 0).astype(np.int64)
            old_inc = inc_matrix[:, i]
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                pts[i] = new_pos
                inc_matrix[:, i] = new_inc
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                
                if time.time() - last_greedy_check > greedy_interval:
                    greedy_lns = _greedy_select_lines(pts, n_lines)
                    greedy_score = _incidence_count(pts, greedy_lns)
                    if greedy_score > best_score:
                        best_score = greedy_score
                        best_pts = pts.copy()
                        best_lns = greedy_lns.copy()
                    last_greedy_check = time.time()
        else:
            # Move line
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines_list[rng.randint(0, len(cand_lines_list) - 1)]
            
            if new_line == old_canon:
                continue
            
            dup = False
            for k in range(n_lines):
                if k != j and _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_line:
                    dup = True
                    break
            if dup:
                continue
            
            A, B, C = new_line
            new_vals = A * pts[:, 0] + B * pts[:, 1] + C
            new_inc = (new_vals == 0).astype(np.int64)
            old_inc = inc_matrix[j, :]
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new_line
                inc_matrix[j, :] = new_inc
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final greedy line selection
    final_lns = _greedy_select_lines(best_pts, n_lines)
    final_score = _incidence_count(best_pts, final_lns)
    
    if final_score >= best_score:
        return best_pts, final_lns
    else:
        return best_pts, best_lns

def _refine_points(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    improved = True
    
    # Get intersection points for preferred locations
    inter_pts = _get_intersection_points(lns.tolist(), max_coord=40)
    inter_pts_list = list(inter_pts)
    
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_pts):
            best_pos = tuple(pts[i])
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[:, i]))
            
            # Try nearby positions and intersection points
            candidates = set()
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    candidates.add((int(pts[i, 0]) + dx, int(pts[i, 1]) + dy))
            if inter_pts_list:
                # Add some random intersection points
                for _ in range(min(20, len(inter_pts_list))):
                    candidates.add(inter_pts_list[random.randint(0, len(inter_pts_list) - 1)])
            
            for new_pos in candidates:
                if new_pos == best_pos:
                    continue
                dup = False
                for k in range(n_pts):
                    if k != i and tuple(pts[k]) == new_pos:
                        dup = True
                        break
                if dup:
                    continue
                
                A = lns[:, 0]
                B = lns[:, 1]
                C = lns[:, 2]
                new_vals = A * new_pos[0] + B * new_pos[1] + C
                new_inc = (new_vals == 0).astype(np.int64)
                new_sum = int(np.sum(new_inc))
                delta = new_sum - old_sum
                
                if delta > best_delta:
                    best_delta = delta
                    best_pos = new_pos
                    best_new_inc = new_inc
            
            if best_delta > 0:
                pts[i] = best_pos
                inc_matrix[:, i] = best_new_inc
                current_score += best_delta
                improved = True
    
    return pts

def _refine_lines(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    improved = True
    
    # Build candidate pool: secants + systematic
    secants = _secant_lines_from_points(pts, max_coeff=10, max_const=50)
    sys_lines = set()
    for A in range(-8, 9):
        for B in range(-8, 9):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 31):
                c = _normalize_line(A, B, C)
                if c:
                    sys_lines.add(c)
    
    cand_lines = set(secants) | sys_lines
    cand_lines_list = list(cand_lines)
    
    while improved and time.time() < deadline - 10:
        improved = False
        for j in range(n_lines):
            best_line = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[j, :]))
            
            for new_line in cand_lines_list:
                if new_line == best_line:
                    continue
                dup = False
                for k in range(n_lines):
                    if k != j and _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_line:
                        dup = True
                        break
                if dup:
                    continue
                
                A, B, C = new_line
                new_vals = A * pts[:, 0] + B * pts[:, 1] + C
                new_inc = (new_vals == 0).astype(np.int64)
                new_sum = int(np.sum(new_inc))
                delta = new_sum - old_sum
                
                if delta > best_delta:
                    best_delta = delta
                    best_line = new_line
                    best_new_inc = new_inc
            
            if best_delta > 0:
                lns[j] = best_line
                inc_matrix[j, :] = best_new_inc
                current_score += best_delta
                improved = True
    
    return lns

def _multi_start_local_search(deadline):
    best_pts = None
    best_lns = None
    best_score = -1
    
    # Start configs
    def _grid_6x9():
        return _initial_grid_diagonal()
    
    def _grid_5x11():
        w, h = 5, 11
        points = []
        for x in range(w):
            for y in range(h):
                points.append((x, y))
        pts = np.array(points[:54], dtype=np.int64)
        grid_lns = _grid_lines(w, h)
        secants = _secant_lines_from_points(pts, max_coeff=5, max_const=20)
        all_cands = grid_lns + secants
        scores = _score_lines_for_points(pts, all_cands)
        sorted_lines = sorted(scores.items(), key=lambda x: -x[1])
        lines = []
        seen = set()
        for line, score in sorted_lines:
            if score < 2:
                break
            canon = _normalize_line(*line)
            if canon and canon not in seen:
                lines.append(canon)
                seen.add(canon)
                if len(lines) == 27:
                    break
        if len(lines) < 27:
            for line in all_cands:
                canon = _normalize_line(*line)
                if canon and canon not in seen:
                    lines.append(canon)
                    seen.add(canon)
                    if len(lines) == 27:
                        break
        return pts, np.array(lines[:27], dtype=np.int64)
    
    def _grid_7x8():
        w, h = 7, 8
        points = []
        for x in range(w):
            for y in range(h):
                points.append((x, y))
        pts = np.array(points[:54], dtype=np.int64)
        grid_lns = _grid_lines(w, h)
        secants = _secant_lines_from_points(pts, max_coeff=5, max_const=20)
        all_cands = grid_lns + secants
        scores = _score_lines_for_points(pts, all_cands)
        sorted_lines = sorted(scores.items(), key=lambda x: -x[1])
        lines = []
        seen = set()
        for line, score in sorted_lines:
            if score < 2:
                break
            canon = _normalize_line(*line)
            if canon and canon not in seen:
                lines.append(canon)
                seen.add(canon)
                if len(lines) == 27:
                    break
        if len(lines) < 27:
            for line in all_cands:
                canon = _normalize_line(*line)
                if canon and canon not in seen:
                    lines.append(canon)
                    seen.add(canon)
                    if len(lines) == 27:
                        break
        return pts, np.array(lines[:27], dtype=np.int64)
    
    def _dense_diagonal():
        points = []
        for i in range(54):
            x = i % 9
            y = i // 9
            points.append((x, y))
        pts = np.array(points, dtype=np.int64)
        grid_lns = _grid_lines(9, 6)
        secants = _secant_lines_from_points(pts, max_coeff=5, max_const=20)
        all_cands = grid_lns + secants
        scores = _score_lines_for_points(pts, all_cands)
        sorted_lines = sorted(scores.items(), key=lambda x: -x[1])
        lines = []
        seen = set()
        for line, score in sorted_lines:
            if score < 2:
                break
            canon = _normalize_line(*line)
            if canon and canon not in seen:
                lines.append(canon)
                seen.add(canon)
                if len(lines) == 27:
                    break
        if len(lines) < 27:
            for line in all_cands:
                canon = _normalize_line(*line)
                if canon and canon not in seen:
                    lines.append(canon)
                    seen.add(canon)
                    if len(lines) == 27:
                        break
        return pts, np.array(lines[:27], dtype=np.int64)
    
    start_configs = [_grid_6x9, _grid_5x11, _grid_7x8, _dense_diagonal]
    
    time_per_config = max(30, (deadline - 40) / len(start_configs))
    
    for cfg_func in start_configs:
        if time.time() > deadline - 40:
            break
        
        cfg_deadline = min(time.time() + time_per_config, deadline - 40)
        try:
            pts, lns = cfg_func()
            
            if time.time() < cfg_deadline - 10:
                pts = _refine_points(pts, lns, cfg_deadline - 10)
            lns = _greedy_select_lines(pts, 27)
            score = _incidence_count(pts, lns)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                best_lns = lns.copy()
            
            if time.time() < cfg_deadline - 5:
                sa_pts, sa_lns = _local_search(pts, lns, cfg_deadline - 5)
                sa_score = _incidence_count(sa_pts, sa_lns)
                if sa_score > best_score:
                    best_score = sa_score
                    best_pts = sa_pts.copy()
                    best_lns = sa_lns.copy()
        except Exception:
            continue
    
    return best_pts, best_lns, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    pts, lns, score = _multi_start_local_search(deadline)
    
    if time.time() < deadline - 15:
        pts = _refine_points(pts, lns, deadline - 15)
        lns = _greedy_select_lines(pts, 27)
    
    if time.time() < deadline - 5:
        lns = _refine_lines(pts, lns, deadline - 5)
    
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
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
