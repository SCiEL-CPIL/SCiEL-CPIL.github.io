import numpy as np
import time
import math
from itertools import product, combinations
from math import gcd

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None  # Invalid
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        val = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(val == 0))
    return total

def _is_valid(points, lines):
    """Validate points and lines per the task contract."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    # Check no NaN/inf (int64 arrays can't have those, but check finite)
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    # No repeated points
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    # No invalid or duplicate lines
    seen_lines = set()
    for l in lns:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _build_6x9():
    """Classical Szemeredi-Trotter k=3 construction: 54 points, 27 lines, 81 incidences."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines.append((a, -1, b))
    lines_arr = np.array(lines, dtype=np.int64)
    return points, lines_arr

def _build_grid6x9():
    """6x9 grid with top-27 lines by incidence count."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    candidates = []
    # Vertical
    for c in range(6):
        candidates.append((1, 0, -c))
    # Horizontal
    for r in range(9):
        candidates.append((0, 1, -r))
    # Diagonal y - x = d
    for d in range(-5, 6):
        candidates.append((-1, 1, d))
    # Anti-diagonal x + y = s
    for s in range(0, 15):
        candidates.append((1, 1, -s))

    # Score each candidate
    scores = []
    for idx, (A, B, C) in enumerate(candidates):
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        scores.append((cnt, idx))
    scores.sort(reverse=True)
    top = [candidates[idx] for _, idx in scores[:27]]
    lines_arr = np.array(top, dtype=np.int64)
    return points, lines_arr

def _build_product_grid():
    """
    Product construction: P = A x B, L = lines through A x B.
    Choose A with |A|=a, B with |B|=b, a*b=54.
    Lines: for each pair (a1,a2) in A and (b1,b2) in B, line through (a1,b1),(a2,b2).
    That's C(a,2)*C(b,2) lines, each with exactly 2 incidences.
    We need exactly 27 lines.
    
    Try a=6, b=9: C(6,2)*C(9,2) = 15*36 = 540 lines, pick top 27.
    Each line has 2 incidences, so 27*2 = 54. Worse than 81.
    
    Alternative: use lines that pass through more points.
    In a 6x9 grid, lines can pass through multiple points.
    Let's just use the grid6x9 approach but with better point selection.
    """
    return _build_grid6x9()

def _build_pencil_grid():
    """
    Pencil construction: points near a line, with many lines through pairs.
    Place 27 points on a line (the x-axis) and 27 points scattered.
    Lines: for each pair of scattered points, the line through them.
    But that gives C(27,2) = 351 lines, each with 2 incidences among scattered points,
    plus potentially hitting axis points.
    
    Better: 13 points on x-axis, 41 elsewhere? No, let's think differently.
    
    Actually, let's try: 27 points on x-axis at x=1..27, y=0.
    27 points at (i, i) for i=1..27? No, those are collinear too.
    
    Let's do: 27 points on x-axis: (1,0), (2,0), ..., (27,0)
    27 points on y-axis: (0,1), (0,2), ..., (0,27)
    Lines: x + y = c for various c. Each line x+y=c hits one x-axis point (c,0) 
    and one y-axis point (0,c) if 1<=c<=27. That's 2 incidences per line.
    27 lines -> 54 incidences. Still less than 81.
    
    Let's stick with the grid approach and try to optimize.
    """
    return _build_grid6x9()

def _build_optimized_grid():
    """
    Try different grid sizes and line selections to maximize incidences.
    Uses V/H/D+/D- families plus steeper slopes for richer candidate pool.
    """
    best_pts = None
    best_lns = None
    best_inc = -1
    
    # Try various grid dimensions
    for (a, b) in [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]:
        xs = np.arange(a)
        ys = np.arange(b)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate candidate lines from multiple families
        candidates = set()
        # Vertical lines: x = c
        for c in range(a):
            candidates.add((1, 0, -c))
        # Horizontal lines: y = r
        for r in range(b):
            candidates.add((0, 1, -r))
        # Diagonals y - x = d (slope 1)
        for d in range(-(a-1), a):
            candidates.add((-1, 1, d))
        # Anti-diagonals x + y = s (slope -1)
        for s in range(0, a + b):
            candidates.add((1, 1, -s))
        # Steeper slopes: y = m*x + c for m in {2, 3, -2, -3}
        for m in [2, 3, -2, -3]:
            # Line: m*x - y + c = 0 -> A=m, B=-1, C=c
            # For point (x,y): m*x - y + c = 0 => c = y - m*x
            # x in [0, a-1], y in [0, b-1]
            # c ranges from -(m*(a-1)) to (b-1) for m>0
            # c ranges from -(b-1) to m*(a-1) for m<0
            if m > 0:
                c_min = -m * (a - 1)
                c_max = b - 1
            else:
                c_min = -(b - 1)
                c_max = -m * (a - 1)
            for c in range(c_min, c_max + 1):
                candidates.add((m, -1, c))
        
        candidates = list(candidates)
        
        # Score and pick top 27
        scores = []
        for idx, (A, B, C) in enumerate(candidates):
            val = A * points[:, 0] + B * points[:, 1] + C
            cnt = int(np.sum(val == 0))
            scores.append((cnt, idx))
        scores.sort(reverse=True)
        top = [candidates[idx] for _, idx in scores[:27]]
        lines_arr = np.array(top, dtype=np.int64)
        
        if _is_valid(points, lines_arr):
            inc = _count_incidences(points, lines_arr)
            if inc > best_inc:
                best_inc = inc
                best_pts, best_lns = points.copy(), lines_arr.copy()
    
    if best_pts is not None:
        return best_pts, best_lns
    return _build_6x9()

def _line_replacement_search(pts, lns, deadline=None):
    """
    Try replacing individual lines with better candidates to increase incidences.
    """
    pts = pts.copy()
    lns = lns.copy()
    current_inc = _count_incidences(pts, lns)
    
    # Generate a larger candidate pool
    a = int(pts[:, 0].max()) + 1
    b = int(pts[:, 1].max()) + 1
    candidates = set()
    for c in range(a):
        candidates.add((1, 0, -c))
    for r in range(b):
        candidates.add((0, 1, -r))
    for d in range(-(a-1), a):
        candidates.add((-1, 1, d))
    for s in range(0, a + b):
        candidates.add((1, 1, -s))
    for m in [2, 3, -2, -3]:
        if m > 0:
            c_min, c_max = -m * (a - 1), b - 1
        else:
            c_min, c_max = -(b - 1), -m * (a - 1)
        for c in range(c_min, c_max + 1):
            candidates.add((m, -1, c))
    
    candidates = list(candidates)
    # Remove current lines from candidates
    current_norms = set()
    for l in lns:
        n = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if n:
            current_norms.add(n)
    
    candidate_list = []
    for c in candidates:
        n = _normalize_line(c[0], c[1], c[2])
        if n and n not in current_norms:
            candidate_list.append(c)
    
    # Pre-compute incidence for each candidate
    cand_incs = []
    for cand in candidate_list:
        A, B, C = cand
        val = A * pts[:, 0] + B * pts[:, 1] + C
        cnt = int(np.sum(val == 0))
        cand_incs.append(cnt)
    
    improved_any = True
    while improved_any:
        if deadline is not None and time.time() > deadline:
            break
        improved_any = False
        for li in range(27):
            if deadline is not None and time.time() > deadline:
                break
            best_gain = 0
            best_cand = None
            for ci, cand in enumerate(candidate_list):
                # Check if this candidate is already in use
                n = _normalize_line(cand[0], cand[1], cand[2])
                if n in current_norms:
                    continue
                # Estimate: remove old line's contribution, add new line's
                # Quick estimate
                gain = cand_incs[ci]
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            
            if best_cand is not None and best_gain > 0:
                # Replace
                old_norm = _normalize_line(int(lns[li, 0]), int(lns[li, 1]), int(lns[li, 2]))
                lns[li] = best_cand
                new_norm = _normalize_line(best_cand[0], best_cand[1], best_cand[2])
                current_norms.discard(old_norm)
                current_norms.add(new_norm)
                current_inc = _count_incidences(pts, lns)
                improved_any = True
                # Update candidate list to exclude newly added line
                # (handled by current_norms check)
    
    return pts, lns, current_inc

def _build_pencil_plus():
    """
    Pencil construction: n/2 points on a line, n/2 points off.
    Lines through pairs of off-line points that also hit the main line.
    """
    # 27 points on x-axis
    axis_pts = [(i, 0) for i in range(1, 28)]
    # 27 points above
    off_pts = [(i, i) for i in range(1, 28)]
    
    all_pts = axis_pts + off_pts
    points = np.array(all_pts, dtype=np.int64)
    
    # Lines through pairs of off_pts
    lines = []
    for i in range(27):
        for j in range(i+1, 27):
            x1, y1 = off_pts[i]
            x2, y2 = off_pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm is not None:
                lines.append(norm)
    
    # Deduplicate and score
    seen = set()
    unique_lines = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            unique_lines.append(l)
    
    # Score each line
    scores = []
    for idx, (A, B, C) in enumerate(unique_lines):
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        scores.append((cnt, idx))
    scores.sort(reverse=True)
    top = [unique_lines[idx] for _, idx in scores[:27]]
    lines_arr = np.array(top, dtype=np.int64)
    
    if _is_valid(points, lines_arr):
        return points, lines_arr
    return _build_6x9()

def _local_search_grid(start_pts, start_lns, max_iters=200, deadline=None):
    """
    Local search: try moving individual points by small integer offsets
    to increase total incidence count.
    """
    pts = start_pts.copy()
    lns = start_lns.copy()
    current_inc = _count_incidences(pts, lns)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for pi in range(54):
            if deadline is not None and time.time() > deadline:
                break
            orig_x, orig_y = int(pts[pi, 0]), int(pts[pi, 1])
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = orig_x + dx, orig_y + dy
                    # Check no duplicate
                    dup = False
                    for pj in range(54):
                        if pj != pi and int(pts[pj, 0]) == new_x and int(pts[pj, 1]) == new_y:
                            dup = True
                            break
                    if dup:
                        continue
                    pts[pi, 0], pts[pi, 1] = new_x, new_y
                    if _is_valid(pts, lns):
                        new_inc = _count_incidences(pts, lns)
                        if new_inc > current_inc:
                            current_inc = new_inc
                            improved = True
                        else:
                            pts[pi, 0], pts[pi, 1] = orig_x, orig_y
            if improved:
                break
        if not improved:
            break
    
    return pts, lns, current_inc

def _build_perturbed_grid():
    """
    Try slightly perturbed grid positions to break symmetry and allow more lines.
    """
    base_pts, base_lns = _build_optimized_grid()
    deadline = time.time() + 120.0
    pts, lns, inc = _local_search_grid(base_pts, base_lns, max_iters=100, deadline=deadline)
    return pts, lns

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    
    candidates = [
        ("sz6x18", _build_6x9),
        ("grid6x9", _build_grid6x9),
        ("opt_grid", _build_optimized_grid),
        ("pencil", _build_pencil_plus),
    ]
    
    best_pts = None
    best_lns = None
    best_inc = -1
    
    for name, fn in candidates:
        if time.time() > deadline:
            break
        try:
            pts, lns = fn()
            if _is_valid(pts, lns):
                inc = _count_incidences(pts, lns)
                if inc > best_inc:
                    best_inc = inc
                    best_pts, best_lns = pts.copy(), lns.copy()
        except Exception:
            continue
    
    # Local search on best result
    if best_pts is not None and time.time() < deadline - 60:
        try:
            pts2, lns2, inc2 = _local_search_grid(best_pts, best_lns, max_iters=150, deadline=deadline - 60)
            if inc2 > best_inc:
                best_inc = inc2
                best_pts, best_lns = pts2, lns2
        except Exception:
            pass
    
    # Line replacement search
    if best_pts is not None and time.time() < deadline - 30:
        try:
            pts3, lns3, inc3 = _line_replacement_search(best_pts, best_lns, deadline=deadline - 30)
            if inc3 > best_inc:
                best_inc = inc3
                best_pts, best_lns = pts3, lns3
        except Exception:
            pass
    
    if best_pts is None:
        pts, lns = _build_6x9()
        return pts, lns
    
    return best_pts, best_lns
