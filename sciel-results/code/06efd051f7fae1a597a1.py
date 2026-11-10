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
    vals = lns[:, 0, None, None] * pts[:, 0, None] + lns[:, 1, None, None] * pts[:, 1, None] + lns[:, 2, None, None]
    return int(np.sum(vals == 0))

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
    if abs(x) > 200 or abs(y) > 200:
        return None
    return (x, y)

def _build_candidate_lines(max_coeff=8, max_const=25):
    """
    Build a diverse pool of lines. We want lines that have many integer intersection points
    with each other. Lines with small coefficients tend to have intersections that are
    more likely to be integers (det is small).
    """
    lines = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_const, max_const + 1):
                canon = _normalize_line(A, B, C)
                if canon:
                    lines.add(canon)
    return list(lines)

def _compute_pairwise_intersections(lines):
    """
    For a set of lines, compute all intersection points and their multiplicity (number of lines passing through).
    Returns a dict: point -> list of line indices
    """
    point_lines = defaultdict(list)
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_lines[pt].append(i)
                point_lines[pt].append(j)
    # This is slightly inefficient because it adds indices multiple times.
    # Let's redo properly.
    point_lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_lines[pt].add(i)
                point_lines[pt].add(j)
    return point_lines

def _generate_lattice_lines_through_points(points, max_slope=4):
    """
    Given a set of points, generate all lines that pass through at least 2 of these points.
    This is a "bottom-up" approach: define points first (or a grid), then find lines.
    """
    lines = set()
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
            # Filter by max_slope (roughly |A/B| or |B/A|)
            if max_slope > 0:
                if A != 0 and B != 0:
                    if abs(A) / abs(B) > max_slope and abs(B) / abs(A) > max_slope:
                        continue
                elif A != 0:
                    if abs(A) > max_slope * abs(B) + max_slope: # crude
                        pass
            canon = _normalize_line(A, B, C)
            if canon:
                # Optional: filter lines with large coefficients
                if abs(canon[0]) <= 10 and abs(canon[1]) <= 10 and abs(canon[2]) <= 50:
                    lines.add(canon)
    return list(lines)

def _evaluate_configuration(points, lines):
    """
    Return incidence count and point->line mapping.
    """
    pts = np.array(points)
    lns = np.array(lines)
    inc_count = 0
    point_lines = defaultdict(set)
    for i, pt in enumerate(pts):
        x, y = pt
        for j, line in enumerate(lns):
            A, B, C = line
            if A * x + B * y + C == 0:
                inc_count += 1
                point_lines[pt].add(j)
    return inc_count, point_lines

def _direct_construction_v1():
    """
    Strategy:
    1. Define a set of 54 points on a grid or specific structure.
    2. Find all lines that pass through at least 2 of these points.
    3. Select the top 27 lines (by number of points they pass through).
    4. This maximizes incidences for the fixed set of points.
    
    To make it stronger, we can try to optimize the point set too.
    But let's start with a good point set. A square grid is good for lines like x=c, y=c, x+y=c, x-y=c.
    
    Let's try a 6x9 grid.
    Points: (x,y) for x in 0..5, y in 0..8. Total 54.
    Lines with many points:
    - x=c: 6 lines, 9 points each.
    - y=c: 9 lines, 6 points each.
    - x+y=c: diagonals. Max points?
      x+y=0: (0,0) -> 1
      x+y=1: (0,1),(1,0) -> 2
      ...
      x+y=4: (0,4)..(4,0) -> 5
      x+y=5: (0,5)..(5,0) -> 6
      x+y=6: (0,6)..(5,1) -> 6 (since x<=5)
      x+y=7: (0,7)..(5,2) -> 6
      x+y=8: (0,8)..(5,3) -> 6
      x+y=9: (1,8)..(5,4) -> 5
      ...
    - x-y=c: similar.
    
    Let's compute the top lines for a 6x9 grid.
    """
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    
    candidate_lines = _generate_lattice_lines_through_points(points, max_slope=3)
    
    # Score each line by number of points it contains
    line_scores = {}
    for line in candidate_lines:
        A, B, C = line
        cnt = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                cnt += 1
        line_scores[line] = cnt
    
    # Sort lines by score descending
    sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
    
    # Select top 27 unique lines
    selected_lines = []
    seen = set()
    for line, score in sorted_lines:
        if score < 2: # A line with 1 point is not useful for incidence maximization (we want multiple points per line)
            break
        if line not in seen:
            selected_lines.append(line)
            seen.add(line)
        if len(selected_lines) == 27:
            break
    
    if len(selected_lines) < 27:
        # Pad with other lines? Or use all 15 grid lines?
        # We have 6+9=15 axis-aligned. Need 12 more.
        # The diagonals x+y=c and x-y=c provide more.
        pass
    
    return points, selected_lines

def _direct_construction_v2():
    """
    Alternative: Use a set of lines first, then optimize points.
    But "direct" suggests building from structure.
    Let's try a different point structure: a triangular or hexagonal lattice might be better for 60-degree lines?
    However, integer coordinates constrain us to slopes p/q.
    
    Let's stick to the grid but maybe a different shape.
    54 = 2 * 3^3.
    Maybe a 3x18 grid? Or 2x27?
    6x9 seems balanced.
    
    Let's refine the line selection.
    For the 6x9 grid:
    Axis aligned:
    x=0..5 (6 lines, 9 pts each) -> Total inc: 54
    y=0..8 (9 lines, 6 pts each) -> Total inc: 54
    Total so far: 108 incidences with 15 lines.
    
    Diagonals x+y=c:
    c=0..13.
    Counts: 1,2,3,4,5,6,6,6,6,5,4,3,2,1
    Top 6 lines: c=5,6,7,8 (6 pts each) -> 4 lines, 24 pts.
    Next: c=4,9 (5 pts each) -> 2 lines, 10 pts.
    
    Diagonals x-y=c:
    c=-8..5.
    Counts: 1,2,3,4,5,6,6,6,6,5,4,3,2,1 (shifted)
    Top 6 lines: c=-4,-3,-2,-1,0,1? No.
    x-y = -8: (0,8) -> 1
    x-y = -7: (0,7),(1,8) -> 2
    ...
    x-y = -3: (0,3)..(2,5)? No.
    Let's check x-y = c for x in 0..5, y in 0..8.
    c = x-y.
    Max c = 5-0 = 5.
    Min c = 0-8 = -8.
    Counts for c:
    c=5: (5,0) -> 1
    c=4: (4,0),(5,1) -> 2
    c=3: (3,0)..(5,2) -> 3
    c=2: (2,0)..(5,3) -> 4
    c=1: (1,0)..(5,4) -> 5
    c=0: (0,0)..(5,5) -> 6
    c=-1: (0,1)..(5,6) -> 6
    c=-2: (0,2)..(5,7) -> 6
    c=-3: (0,3)..(5,8) -> 6
    c=-4: (0,4)..(4,8) -> 5
    c=-5: (0,5)..(3,8) -> 4
    ...
    
    So we have 4 lines with 6 pts for x+y, and 4 lines with 6 pts for x-y.
    Total 8 "full" diagonal lines (6 pts each).
    
    If we take:
    6 vertical (9 pts)
    9 horizontal (6 pts)
    4 diag1 (6 pts)
    4 diag2 (6 pts)
    Total lines: 6+9+4+4 = 23 lines.
    Incidences: 6*9 + 9*6 + 4*6 + 4*6 = 54 + 54 + 24 + 24 = 156.
    
    We need 27 lines. We have 4 slots left.
    Next best lines:
    x+y=4 (5 pts), x+y=9 (5 pts)
    x-y=-4 (5 pts), x-y=1 (5 pts)
    Add these 4.
    Incidences added: 4 * 5 = 20.
    Total incidences: 156 + 20 = 176.
    
    This is a strong baseline.
    
    Can we do better?
    What if we sacrifice some horizontal lines (6 pts) to add more diagonals?
    Horizontal lines have 6 pts. The "next" diagonals have 5 pts.
    So keeping horizontals is better.
    
    What if we change the grid to make diagonals denser?
    A square grid N x N has diagonals of length N.
    But we need 54 points. 7x7=49, 8x8=64.
    If we use 8x8, we have 64 points, but can only use 54.
    If we remove 10 points from corners, diagonals break.
    
    Let's stick with 6x9.
    
    Is there a better arrangement?
    Consider lines of slope 1/2 or 2?
    In 6x9, lines with slope 1/2: y - 0.5x = c -> 2y - x = c.
    x in 0..5, y in 0..8.
    2y - x = c.
    Max c = 16 - 0 = 16.
    Min c = 0 - 5 = -5.
    Counts?
    It might be similar to slope 1.
    
    The current score is 2.23. 176 incidences.
    The seed likely has ~150?
    Let's implement this direct construction and then run local search to improve.
    """
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    
    candidate_lines = _generate_lattice_lines_through_points(points, max_slope=3)
    
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
    
    return points, selected_lines

def _local_search(points, lines, deadline):
    """
    Fast local search to improve the direct construction.
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
    
    # Build set of current points for fast lookup
    current_point_set = set(map(tuple, pts.tolist()))
    
    # Candidate points: intersections of current lines + neighbors
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    
    # Add current points
    for p in current_point_set:
        cand_points.add(p)
    
    # Candidate lines: lines through pairs of current points
    cand_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 15 and abs(B) <= 15 and abs(C) <= 60:
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.add(canon)
    
    # Add current lines
    for j in range(n_lines):
        canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
        if canon:
            cand_lines.add(canon)
    
    # Add some random lines
    for _ in range(300):
        A = rng.randint(-8, 8)
        B = rng.randint(-8, 8)
        if A == 0 and B == 0:
            continue
        C = rng.randint(-30, 30)
        canon = _normalize_line(A, B, C)
        if canon:
            cand_lines.add(canon)
            
    cand_points = list(cand_points)
    cand_lines = list(cand_lines)
    
    if not cand_points or not cand_lines:
        return best_pts, best_lns, best_total
    
    # Precompute canonical forms for cand_lines
    cand_lines_canon = []
    for cl in cand_lines:
        canon = _normalize_line(cl[0], cl[1], cl[2])
        if canon:
            cand_lines_canon.append(canon)
    cand_lines = cand_lines_canon
    
    # Precompute line coefficients as numpy arrays for faster computation
    lns_A = lns[:, 0].astype(np.int64)
    lns_B = lns[:, 1].astype(np.int64)
    lns_C = lns[:, 2].astype(np.int64)
    
    T_start = 5.0
    T_end = 0.01
    n_iters = 100000
    iter_count = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        if rng.random() < 0.5:
            # Point move
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            
            if new_pos == old_pos:
                continue
            if new_pos in current_point_set:
                # Check if it's the same point
                if new_pos == old_pos:
                    continue
                # Find if another point is at this position
                found = False
                for k in range(n_pts):
                    if k != i and tuple(pts[k]) == new_pos:
                        found = True
                        break
                if found:
                    continue
            
            x_new, y_new = new_pos
            # Compute delta using vectorized ops
            vals = lns_A * x_new + lns_B * y_new + lns_C
            now_inc = (vals == 0).astype(int)
            was_inc = inc[i, :].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or (delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                total_inc = new_total
                pts[i] = new_pos
                current_point_set.discard(old_pos)
                current_point_set.add(new_pos)
                inc[i, :] = now_inc.astype(bool)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # Line move
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines[rng.randint(0, len(cand_lines) - 1)]
            new_canon = _normalize_line(new_line[0], new_line[1], new_line[2])
            
            if new_canon is None or new_canon == old_canon:
                continue
            # Check for duplicate
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    k_canon = _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2]))
                    if k_canon == new_canon:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            A, B, C = new_line
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            now_inc = (vals == 0).astype(int)
            was_inc = inc[:, j].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or (delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                total_inc = new_total
                lns[j] = new_line
                inc[:, j] = now_inc.astype(bool)
                # Update lns_A, lns_B, lns_C
                lns_A[j] = A
                lns_B[j] = B
                lns_C[j] = C
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
    return best_pts, best_lns, best_total

def _polish_pass(points, lines, deadline):
    """
    A final polish pass that tries to improve each point and line individually.
    """
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
    
    # For each point, try all candidate positions and pick the best
    current_point_set = set(map(tuple, pts.tolist()))
    
    # Generate candidate points
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    for p in current_point_set:
        cand_points.add(p)
    cand_points = list(cand_points)
    
    # For each point, find best position
    for i in range(n_pts):
        if time.time() > deadline:
            break
        old_pos = tuple(pts[i])
        best_delta = 0
        best_new_pos = old_pos
        
        for new_pos in cand_points:
            if new_pos == old_pos:
                continue
            # Check if new_pos is already used by another point
            if new_pos in current_point_set:
                if new_pos != old_pos:
                    continue
            
            x_new, y_new = new_pos
            vals = lns[:, 0] * x_new + lns[:, 1] * y_new + lns[:, 2]
            now_inc = (vals == 0).astype(int)
            was_inc = inc[i, :].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            if delta > best_delta:
                best_delta = delta
                best_new_pos = new_pos
        
        if best_delta > 0:
            pts[i] = best_new_pos
            current_point_set.discard(old_pos)
            current_point_set.add(best_new_pos)
            x_new, y_new = best_new_pos
            vals = lns[:, 0] * x_new + lns[:, 1] * y_new + lns[:, 2]
            inc[i, :] = (vals == 0)
            total_inc += best_delta
    
    # For each line, try to improve it
    cand_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 15 and abs(B) <= 15 and abs(C) <= 60:
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.add(canon)
    for j in range(n_lines):
        canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
        if canon:
            cand_lines.add(canon)
    cand_lines = list(cand_lines)
    
    for j in range(n_lines):
        if time.time() > deadline:
            break
        old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
        best_delta = 0
        best_new_line = None
        
        for new_line in cand_lines:
            new_canon = _normalize_line(new_line[0], new_line[1], new_line[2])
            if new_canon is None or new_canon == old_canon:
                continue
            # Check for duplicate
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    k_canon = _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2]))
                    if k_canon == new_canon:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            A, B, C = new_line
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            now_inc = (vals == 0).astype(int)
            was_inc = inc[:, j].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            if delta > best_delta:
                best_delta = delta
                best_new_line = new_line
        
        if best_delta > 0 and best_new_line is not None:
            lns[j] = best_new_line
            A, B, C = best_new_line
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            inc[:, j] = (vals == 0)
            total_inc += best_delta
    
    return pts, lns, total_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # Direct construction - try multiple grids
    pts, lns = _direct_construction_v2()
    
    # Ensure we have exactly 27 lines
    if len(lns) < 27:
        rng = random.Random(1)
        while len(lns) < 27:
            A = rng.randint(-5, 5)
            B = rng.randint(-5, 5)
            if A == 0 and B == 0:
                continue
            C = rng.randint(-20, 20)
            canon = _normalize_line(A, B, C)
            if canon and canon not in set(lns):
                lns.append(canon)
    else:
        lns = lns[:27]
        
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    # Local search
    if time.time() < deadline - 60:
        ls_pts, ls_lns, ls_count = _local_search(pts, lns, deadline - 30)
        pts = ls_pts
        lns = ls_lns
    
    # Polish pass
    if time.time() < deadline - 15:
        pp_pts, pp_lns, pp_count = _polish_pass(pts, lns, deadline - 5)
        pts = pp_pts
        lns = pp_lns
    
    # Final validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback: simple grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
        lns = lines
    
    return pts, lns
