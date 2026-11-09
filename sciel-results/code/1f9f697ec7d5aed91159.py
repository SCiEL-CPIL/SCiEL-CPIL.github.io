# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd


def _normalize_line(A, B, C):
    """Normalize (A,B,C) to canonical form."""
    if A == 0 and B == 0 and C == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _is_valid(points, lines):
    """Check validity of (points, lines)."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if np.any(np.abs(points - np.round(points)) > 1e-6):
        return False
    if np.any(np.abs(lines - np.round(lines)) > 1e-6):
        return False
    pts = points.astype(np.int64)
    if len(set(map(tuple, pts.tolist()))) != 54:
        return False
    lns = lines.astype(np.int64)
    for row in lns:
        if row[0] == 0 and row[1] == 0:
            return False
    canon = set()
    for row in lns:
        c = _normalize_line(int(row[0]), int(row[1]), int(row[2]))
        if c is None:
            return False
        if c in canon:
            return False
        canon.add(c)
    return True


def _count_incidences(points, lines):
    """Count point-line incidences exactly in int64."""
    pts = points.astype(np.int64)
    lns = lines.astype(np.int64)
    # Vectorized: compute all 54x27 values at once
    # val[i,j] = A_j * x_i + B_j * y_i + C_j
    val = lns[0] * pts[:, 0:1] + lns[1] * pts[:, 1:2] + lns[2]
    return int(np.sum(val == 0))


def _seed_construction():
    """The original seed: 3x9 grid with lines y=a*x+b."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines


def _build_incidence_matrix(points, lines):
    """Return (54, 27) boolean matrix where M[i,j]=True if point i is on line j."""
    pts = points.astype(np.int64)
    lns = lines.astype(np.int64)
    val = lns[0] * pts[:, 0:1] + lns[1] * pts[:, 1:2] + lns[2]
    return (val == 0)


def _greedy_line_search(points, n_lines=27, max_iters=500, deadline=None):
    """
    Given fixed points, greedily find n_lines lines that maximize incidences.
    Strategy: enumerate candidate lines from pairs of points, score them,
    then do local search swapping lines.
    """
    pts = points.astype(np.int64)
    n_pts = len(pts)
    
    # Build incidence matrix for current lines
    if len(points) < 2:
        return None, -1
    
    # Generate candidate lines from pairs of points
    # For efficiency, use a subset of pairs
    candidate_lines = {}
    n = n_pts
    # Sample pairs to keep it manageable
    if n <= 54:
        # All pairs
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                A = dy
                B = -dx
                C = x1 * y2 - x2 * y1
                c = _normalize_line(A, B, C)
                if c is None:
                    continue
                candidate_lines[c] = True
    
    # Also add horizontal and vertical lines through each point
    for i in range(n):
        x1, y1 = pts[i]
        # Horizontal: y = y1 => 0*x + 1*y + (-y1) = 0
        c = _normalize_line(0, 1, -y1)
        if c:
            candidate_lines[c] = True
        # Vertical: x = x1 => 1*x + 0*y + (-x1) = 0
        c = _normalize_line(1, 0, -x1)
        if c:
            candidate_lines[c] = True
    
    cand_list = list(candidate_lines.keys())
    if len(cand_list) == 0:
        return None, -1
    
    # Score each candidate line: how many points it hits
    line_scores = {}
    for c in cand_list:
        A, B, C = c
        val = A * pts[:, 0] + B * pts[:, 1] + C
        line_scores[c] = int(np.sum(val == 0))
    
    # Sort by score descending
    sorted_cands = sorted(cand_list, key=lambda c: -line_scores[c])
    
    # Greedy selection: pick top lines, but avoid duplicates (already unique)
    # and ensure we get distinct lines
    selected = []
    for c in sorted_cands:
        if len(selected) >= n_lines:
            break
        selected.append(c)
    
    if len(selected) < n_lines:
        return None, -1
    
    lines = np.array(selected, dtype=np.int64)
    M = _build_incidence_matrix(points, lines)
    current_score = int(np.sum(M))
    
    # Local search: try swapping one line with another
    best_score = current_score
    best_lines = lines.copy()
    
    used_canons = set(selected)
    available = [c for c in sorted_cands if c not in used_canons]
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for idx in range(n_lines):
            if deadline is not None and time.time() > deadline:
                break
            # Try replacing line idx with each available line
            for alt in available[:200]:  # limit to top 200 alternatives
                if alt in used_canons:
                    continue
                # Compute delta: remove old line, add new line
                old_inc = int(np.sum(M[:, idx]))
                new_A, new_B, new_C = alt
                new_val = new_A * pts[:, 0] + new_B * pts[:, 1] + new_C
                new_inc = int(np.sum(new_val == 0))
                delta = new_inc - old_inc
                if delta > 0:
                    # Apply swap
                    M[:, idx] = (new_val == 0)
                    lines[idx] = alt
                    used_canons.discard(selected[idx])
                    used_canons.add(alt)
                    selected[idx] = alt
                    current_score += delta
                    best_score = current_score
                    best_lines = lines.copy()
                    improved = True
                    break  # restart scan after improvement
        
        if not improved:
            break
    
    return best_lines, best_score


def _greedy_point_search(points, lines, max_iters=300, deadline=None):
    """
    Given fixed lines, try to improve by swapping points.
    For each point, try moving it to a position that lies on more lines.
    """
    lns = lines.astype(np.int64)
    pts = points.astype(np.int64)
    M = _build_incidence_matrix(points, lines)
    current_score = int(np.sum(M))
    
    # For each point position, find candidate new positions
    # A good new position is one that lies on many of the 27 lines
    # Generate candidates: intersections of pairs of lines, plus existing point positions
    
    # Compute all pairwise line intersections
    candidates = set()
    # Add current points
    for i in range(len(pts)):
        candidates.add((int(pts[i, 0]), int(pts[i, 1])))
    
    # Add intersections of pairs of lines
    for i in range(len(lns)):
        for j in range(i + 1, len(lns)):
            A1, B1, C1 = lns[i]
            A2, B2, C2 = lns[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue  # parallel
            # x = (B1*C2 - B2*C1) / det
            # y = (A2*C1 - A1*C2) / det
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            if num_x % det == 0 and num_y % det == 0:
                x = num_x // det
                y = num_y // det
                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                    candidates.add((x, y))
    
    # For each candidate, compute how many lines it's on
    cand_list = list(candidates)
    cand_scores = []
    for (x, y) in cand_list:
        val = lns[0] * x + lns[1] * y + lns[2]
        cnt = int(np.sum(val == 0))
        cand_scores.append((cnt, x, y))
    
    cand_scores.sort(reverse=True)
    
    best_score = current_score
    best_points = pts.copy()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(len(pts)):
            if deadline is not None and time.time() > deadline:
                break
            old_inc = int(M[i])
            # Try best candidates that aren't already used
            for cnt, x, y in cand_scores[:100]:
                if cnt <= old_inc:
                    break
                # Check if this position is already used by another point
                new_pt = np.array([x, y])
                if np.any((pts == new_pt).all(axis=1)):
                    continue
                # Apply
                pts[i] = new_pt
                # Update M row i
                val = lns[0] * x + lns[1] * y + lns[2]
                M[i] = (val == 0)
                current_score += cnt - old_inc
                best_score = current_score
                best_points = pts.copy()
                improved = True
                break
        
        if not improved:
            break
    
    return best_points, best_score


def _full_local_search(points, lines, deadline):
    """
    Alternating greedy point and line search.
    """
    pts = points.copy()
    lns = lines.copy()
    current_score = _count_incidences(pts, lns)
    
    for round_num in range(20):
        if time.time() > deadline:
            break
        
        # Improve lines
        new_lines, line_score = _greedy_line_search(pts, n_lines=27, max_iters=100, deadline=deadline)
        if new_lines is not None and line_score > current_score:
            lns = new_lines
            current_score = line_score
        
        # Improve points
        new_pts, pt_score = _greedy_point_search(pts, lns, max_iters=100, deadline=deadline)
        if pt_score > current_score:
            pts = new_pts
            current_score = pt_score
        
        if time.time() > deadline:
            break
    
    return pts, lns, current_score


def _grid_search_construction(deadline):
    """
    Try multiple grid configurations and for each, run local search.
    """
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try various grid shapes
    grids = [
        (3, 18), (18, 3), (6, 9), (9, 6), (2, 27), (27, 2),
        (4, 14), (14, 4), (5, 11), (11, 5), (7, 8), (8, 7),
        (1, 54), (54, 1), (2, 28), (28, 2), (3, 19), (19, 3),
    ]
    
    for gw, gh in grids:
        if time.time() > deadline:
            break
        if gw * gh < 54:
            continue
        if gw * gh > 300:
            continue
        
        xs = np.arange(gw)
        ys = np.arange(gh)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Select 54 points: take first 54 or a strategic subset
        n_all = len(all_points)
        if n_all >= 54:
            # Take a spread-out subset
            indices = np.linspace(0, n_all - 1, 54, dtype=int)
            points = all_points[indices]
        else:
            points = all_points
        
        # Initial lines: find best 27 lines for these points
        lines, score = _greedy_line_search(points, n_lines=27, max_iters=50, deadline=deadline)
        if lines is None:
            # Fallback: use seed-style lines
            lines = _seed_construction()[1]
            score = _count_incidences(points, lines)
        
        # Local search
        pts, lns, final_score = _full_local_search(points, lines, deadline)
        
        if final_score > best_score:
            best_score = final_score
            best_points = pts.copy()
            best_lines = lns.copy()
    
    return best_points, best_lines, best_score


def _structured_construction(deadline):
    """
    Use structured algebraic constructions known to have high incidences.
    e.g., points on a parabola with tangent lines, or grid with specific line families.
    """
    best_points = None
    best_lines = None
    best_score = -1
    
    # Construction 1: Points on y = x^2 mod p, lines are tangents
    # For p=17, we get 17 points; need 54, so use 3 parabolas or larger p
    # Actually let's use: points (i, i^2) for i in range(54), lines through pairs
    
    # Construction 2: 6x9 grid with all horizontal, vertical, and diagonal lines
    for gw, gh in [(6, 9), (9, 6), (3, 18), (18, 3)]:
        if time.time() > deadline:
            break
        
        xs = np.arange(gw)
        ys = np.arange(gh)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Take all 54 points if exactly 54
        if len(all_points) == 54:
            points = all_points
        elif len(all_points) > 54:
            indices = np.linspace(0, len(all_points) - 1, 54, dtype=int)
            points = all_points[indices]
        else:
            continue
        
        # Generate structured lines
        line_set = set()
        # Horizontal lines
        for y in range(gh):
            c = _normalize_line(0, 1, -y)
            if c:
                line_set.add(c)
        # Vertical lines
        for x in range(gw):
            c = _normalize_line(1, 0, -x)
            if c:
                line_set.add(c)
        # Diagonal lines (slope 1 and -1)
        for i in range(gw):
            for j in range(gh):
                # slope 1: x - y = const
                c = _normalize_line(1, -1, -(i - j))
                if c:
                    line_set.add(c)
                # slope -1: x + y = const
                c = _normalize_line(1, 1, -(i + j))
                if c:
                    line_set.add(c)
        
        # Score lines
        pts = points.astype(np.int64)
        scored = []
        for c in line_set:
            A, B, C = c
            val = A * pts[:, 0] + B * pts[:, 1] + C
            cnt = int(np.sum(val == 0))
            scored.append((cnt, c))
        
        scored.sort(reverse=True)
        
        # Pick top 27
        selected = [c for _, c in scored[:27]]
        if len(selected) < 27:
            continue
        
        lines = np.array(selected, dtype=np.int64)
        score = _count_incidences(points, lines)
        
        # Local search
        pts, lns, final_score = _full_local_search(points, lines, deadline)
        
        if final_score > best_score:
            best_score = final_score
            best_points = pts.copy()
            best_lines = lns.copy()
    
    return best_points, best_lines, best_score


def _random_restart_search(deadline, n_restarts=50):
    """
    Random restarts: generate random point sets and optimize lines.
    """
    best_points = None
    best_lines = None
    best_score = -1
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(r * 12345 + 678)
        
        # Random points in a small box
        box = np.random.choice([5, 8, 10, 12, 15, 20], p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
        pts = np.random.randint(0, box, size=(54, 2))
        
        # Ensure uniqueness
        unique_pts = []
        seen = set()
        for p in pts:
            t = (int(p[0]), int(p[1]))
            if t not in seen:
                seen.add(t)
                unique_pts.append(p)
        
        while len(unique_pts) < 54:
            new_p = np.random.randint(0, box + 5, size=2)
            t = (int(new_p[0]), int(new_p[1]))
            if t not in seen:
                seen.add(t)
                unique_pts.append(new_p)
        
        points = np.array(unique_pts[:54], dtype=np.int64)
        
        # Find best lines
        lines, score = _greedy_line_search(points, n_lines=27, max_iters=50, deadline=deadline)
        if lines is None:
            continue
        
        # Local search
        pts, lns, final_score = _full_local_search(points, lines, deadline)
        
        if final_score > best_score:
            best_score = final_score
            best_points = pts.copy()
            best_lines = lns.copy()
    
    return best_points, best_lines, best_score


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.
    
    Uses vectorized incidence computation + greedy local search.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    # Start with the seed construction
    best_points, best_lines = _seed_construction()
    best_score = _count_incidences(best_points, best_lines)
    
    # Try structured construction
    if time.time() < deadline - 30:
        pts, lns, score = _structured_construction(deadline)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
    
    # Try grid search construction
    if time.time() < deadline - 30:
        pts, lns, score = _grid_search_construction(deadline)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
    
    # Random restarts
    if time.time() < deadline - 10:
        pts, lns, score = _random_restart_search(deadline, n_restarts=100)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
    
    # Final local search on the best found
    if time.time() < deadline - 5:
        pts, lns, score = _full_local_search(best_points, best_lines, deadline)
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
    
    # Validate before returning
    if not _is_valid(best_points, best_lines):
        # Fallback to seed
        best_points, best_lines = _seed_construction()
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
