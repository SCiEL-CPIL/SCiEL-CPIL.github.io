"""Structural rewrite: model as maximum-weight clique over 4-point line candidates
using scipy MILP, with a fast local-search fallback."""
import time
import random
from math import gcd
from collections import defaultdict

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _line_key(a, b):
    x1, y1 = a
    x2, y2 = b
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_lines(points):
    """Return (exact4, fiveplus, line_map) where line_map: key -> set of point indices."""
    n = len(points)
    line_pts = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            line_pts[k].add(i)
            line_pts[k].add(j)
    exact4 = 0
    fiveplus = 0
    for s in line_pts.values():
        if len(s) == 4:
            exact4 += 1
        elif len(s) >= 5:
            fiveplus += 1
    return exact4, fiveplus, line_pts

def _score(points):
    e4, f5, _ = _count_lines(points)
    if f5 > 0:
        return -10**9 - f5
    return e4

def _valid(points):
    if len(set(map(tuple, points))) != len(points):
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, f5, _ = _count_lines(points)
    return f5 == 0

# --- MILP-based clique search over 4-point line candidates ---

def _generate_line_candidates(points, max_candidates=2000):
    """Generate candidate 4-point lines from the current configuration.
    For each pair of points, extend to a line and see if exactly 4 points lie on it.
    Also try random 4-subsets to find lines not present in current config."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    candidates = []  # list of frozenset of point indices
    seen = set()

    # Lines determined by pairs in current config
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            # Check how many of the 16 points are on this line
            on_line = set()
            for idx in range(n):
                A, B, C = k
                if A * points[idx][0] + B * points[idx][1] == C:
                    on_line.add(idx)
            if len(on_line) == 4:
                key = frozenset(on_line)
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)

    # Random 4-subsets: check if they are collinear
    rng_local = random.Random(42)
    for _ in range(3000):
        subset = set(rng_local.sample(range(n), 4))
        # Check collinearity
        p0, p1, p2, p3 = [points[i] for i in sorted(subset)]
        # Cross product (p1-p0) x (p2-p0) == 0 and (p3-p0)
        ax, ay = p1[0]-p0[0], p1[1]-p0[1]
        bx, by = p2[0]-p0[0], p2[1]-p0[1]
        cx, cy = p3[0]-p0[0], p3[1]-p0[1]
        if ax*by - ay*bx == 0 and ax*cy - ay*cx == 0:
            key = frozenset(subset)
            if key not in seen:
                seen.add(key)
                candidates.append(key)

    return candidates[:max_candidates]

def _milp_select_lines(points, candidates, time_limit=30):
    """Use MILP to select a set of 4-point lines that are pairwise compatible
    (share at most 1 point) and maximize count."""
    m = len(candidates)
    if m == 0:
        return 0, []

    # Compatibility: two lines are compatible if they share at most 1 point
    # Build conflict matrix
    conflict = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        for j in range(i+1, m):
            overlap = len(candidates[i] & candidates[j])
            if overlap > 1:
                conflict[i, j] = 1
                conflict[j, i] = 1

    # Objective: maximize sum of selected lines
    c = np.ones(m, dtype=np.float64)

    # Constraints: x_i + x_j <= 1 for conflicting pairs
    # Also: each point can be in at most... well, we want to maximize lines,
    # and each line uses 4 distinct points. Points can be shared across lines (up to 2 per pair of lines).
    # The conflict constraint handles the "share at most 1" rule.

    A_rows = []
    b_lower = []
    b_upper = []
    for i in range(m):
        for j in range(i+1, m):
            if conflict[i, j]:
                row = np.zeros(m, dtype=np.float64)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(1.0)

    if A_rows:
        A = np.array(A_rows, dtype=np.float64)
        lb = np.array(b_lower, dtype=np.float64)
        ub = np.array(b_upper, dtype=np.float64)
        constraints = LinearConstraint(A, lb, ub)
    else:
        constraints = LinearConstraint(np.zeros((0, m)), -np.inf, np.inf)

    bounds = Bounds(0, 1, shape=m)
    integrality = np.ones(m, dtype=np.int8)

    t0 = time.time()
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            selected = [i for i in range(m) if res.x[i] > 0.5]
            return len(selected), selected
    except Exception:
        pass

    # Fallback: greedy
    selected = []
    used = [False] * m
    # Sort by... just take all non-conflicting greedily
    order = sorted(range(m), key=lambda i: -len(candidates[i]))
    for i in order:
        if used[i]:
            continue
        ok = True
        for j in selected:
            if conflict[i, j]:
                ok = False
                break
        if ok:
            selected.append(i)
            used[i] = True

    return len(selected), selected

def _reconstruct_from_lines(points, selected_lines, candidates, rng, max_iters=500):
    """Given a set of 4-point lines, try to find a point configuration that realizes them.
    This is a constraint satisfaction problem. We use local search:
    Start from current points, and for each selected line, try to move points onto it."""
    n = len(points)
    pts = [list(p) for p in points]

    # For each selected line, we know which 4 point indices should be on it.
    # We need to find integer coordinates for those 4 points that are collinear
    # and consistent across lines (a point index appears in at most... well, lines share at most 1 point).

    # Actually, let's use a different approach: local search that rewards being on selected lines.
    # Score = (number of selected lines fully realized) - (violations)

    selected_sets = [candidates[i] for i in selected_lines]

    def _line_score(pts):
        """Count how many selected lines are fully realized."""
        count = 0
        for s in selected_sets:
            idxs = sorted(s)
            p0, p1, p2, p3 = [pts[i] for i in idxs]
            ax, ay = p1[0]-p0[0], p1[1]-p0[1]
            bx, by = p2[0]-p0[0], p2[1]-p0[1]
            cx, cy = p3[0]-p0[0], p3[1]-p0[1]
            if ax*by - ay*bx == 0 and ax*cy - ay*cx == 0:
                count += 1
        return count

    cur_line_score = _line_score(pts)
    cur_total = _score(pts)

    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        best_move = None
        best_val = cur_total
        # Try moving each point
        for idx in range(n):
            x0, y0 = pts[idx]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check distinctness
                    if any(pts[j][0] == nx and pts[j][1] == ny for j in range(n) if j != idx):
                        continue
                    old = pts[idx]
                    pts[idx] = [nx, ny]
                    ls = _line_score(pts)
                    v = _score(pts)
                    # Prefer moves that increase line score, then total score
                    if (ls, v) > (cur_line_score, cur_total) or (ls == cur_line_score and v > cur_total):
                        best_val = v
                        best_move = (idx, [nx, ny])
                        cur_line_score = ls
                        cur_total = v
                    pts[idx] = old

        if best_move is not None:
            idx, new_pt = best_move
            pts[idx] = new_pt
            cur_total = best_val
        else:
            break

    return pts, cur_total

# --- Local search (improved) ---

def _local_search(points, rng, deadline_fn):
    pts = [list(p) for p in points]
    cur = _score(pts)
    improved = True
    stall = 0

    while improved and deadline_fn() > 5.0:
        improved = False
        n = len(pts)
        order = list(range(n))
        rng.shuffle(order)

        for idx in order:
            if deadline_fn() < 5.0:
                return pts, cur
            best_new = None
            best_val = cur
            x0, y0 = pts[idx]

            # Generate diverse candidates
            cands = set()
            # Small moves
            for _ in range(30):
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    cands.add((nx, ny))
            # Medium moves
            for _ in range(15):
                dx = rng.randint(-15, 15)
                dy = rng.randint(-15, 15)
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    cands.add((nx, ny))
            # Large jumps
            for _ in range(5):
                dx = rng.randint(-50, 50)
                dy = rng.randint(-50, 50)
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    cands.add((nx, ny))

            for c in cands:
                if any(pts[j][0] == c[0] and pts[j][1] == c[1] for j in range(n) if j != idx):
                    continue
                old = pts[idx]
                pts[idx] = list(c)
                v = _score(pts)
                if v > best_val:
                    best_val = v
                    best_new = list(c)
                pts[idx] = old

            if best_new is not None:
                pts[idx] = best_new
                cur = best_val
                improved = True
                stall = 0
            else:
                stall += 1

        if stall > 3:
            break

    return pts, cur

def _seed_grid(step, ox, oy, s=4):
    return [[ox + i * step, oy + j * step] for i in range(s) for j in range(s)]

def _seed_affine_grid(rng):
    base = [[i, j] for i in range(4) for j in range(4)]
    while True:
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-3, 3)
        d = rng.randint(-3, 3)
        det = a * d - b * c
        if det != 0:
            break
    ox = rng.randint(-50, 50)
    oy = rng.randint(-50, 50)
    pts = []
    for (x, y) in base:
        nx = a * x + b * y + ox
        ny = c * x + d * y + oy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            pts.append([nx, ny])
    return pts

def _seed_perturbed(rng):
    base = _seed_grid(1, 0, 0)
    return [[x + rng.randint(-2, 2), y + rng.randint(-2, 2)] for x, y in base]

def construct():
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = _START + 145.0  # safety margin for 150s limit
    rng = random.Random(12345)

    seeds = []
    seeds.append(_seed_grid(1, 0, 0))
    for step in (2, 3, 5, 7):
        seeds.append(_seed_grid(step, 0, 0))
    seeds.append(_seed_grid(1, 3, 5))
    for _ in range(20):
        s = _seed_affine_grid(rng)
        if len(s) == 16:
            seeds.append(s)
    for _ in range(20):
        seeds.append(_seed_perturbed(rng))

    best_points = None
    best_score = -10**12

    # Phase 1: Quick local search from seeds
    for s in seeds:
        if _remaining() < 10.0:
            break
        if len(set(map(tuple, s))) != 16:
            continue
        if any(abs(x) > 1000 or abs(y) > 1000 for x, y in s):
            continue
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_points = [list(p) for p in s]
        if _remaining() < 15.0:
            break
        pts, sc2 = _local_search(s, rng, _remaining)
        if sc2 > best_score:
            best_score = sc2
            best_points = pts

    # Phase 2: MILP-guided search from best configuration
    if best_points is not None and _remaining() > 20.0:
        candidates = _generate_line_candidates(best_points, max_candidates=500)
        if len(candidates) > 0:
            n_sel, sel_idx = _milp_select_lines(best_points, candidates, time_limit=20)
            if _remaining() > 10.0 and n_sel > 0:
                pts2, sc2 = _reconstruct_from_lines(best_points, sel_idx, candidates, rng, max_iters=300)
                if sc2 > best_score:
                    best_score = sc2
                    best_points = pts2

    # Phase 3: More local search from the best
    if best_points is not None and _remaining() > 10.0:
        pts3, sc3 = _local_search(best_points, rng, _remaining)
        if sc3 > best_score:
            best_score = sc3
            best_points = pts3

    # Phase 4: Second round of MILP from updated best
    if best_points is not None and _remaining() > 15.0:
        candidates2 = _generate_line_candidates(best_points, max_candidates=800)
        if len(candidates2) > 0:
            n_sel2, sel_idx2 = _milp_select_lines(best_points, candidates2, time_limit=15)
            if _remaining() > 8.0 and n_sel2 > 0:
                pts4, sc4 = _reconstruct_from_lines(best_points, sel_idx2, candidates2, rng, max_iters=200)
                if sc4 > best_score:
                    best_score = sc4
                    best_points = pts4

    # Phase 5: Final local search
    if best_points is not None and _remaining() > 5.0:
        pts5, sc5 = _local_search(best_points, rng, _remaining)
        if sc5 > best_score:
            best_score = sc5
            best_points = pts5

    # Safety fallback
    if best_points is None or not _valid(best_points):
        best_points = [[x, y] for x in range(4) for y in range(4)]

    return [[int(x), int(y)] for x, y in best_points]
