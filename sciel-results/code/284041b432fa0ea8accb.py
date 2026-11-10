# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import gcd
from itertools import combinations

_T0 = time.time()
_DEADLINE = 620.0  # safety margin vs 630s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _canon(A, B, C):
    """Canonical form: divide by gcd, make first nonzero coefficient positive."""
    g = gcd(abs(int(A)), gcd(abs(int(B)), abs(int(C))))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _evaluate(points, lines):
    """Exact incidence count in int64."""
    total = 0
    for (px, py) in points:
        for (A, B, C) in lines:
            if A * px + B * py + C == 0:
                total += 1
    return total


def _compute_all_lines(points):
    """Compute all distinct lines through pairs of points with their incidence counts."""
    P = list(points)
    n = len(P)
    idx = np.array(P, dtype=np.int64)
    line_info = {}  # canon -> (A_, B_, C_, cnt)
    
    for i in range(n):
        x1, y1 = P[i]
        for j in range(i + 1, n):
            x2, y2 = P[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            ct = _canon(A, B, C)
            if ct not in line_info:
                A_, B_, C_ = ct
                mask = (A_ * idx[:, 0] + B_ * idx[:, 1] + C_ == 0)
                cnt = int(np.sum(mask))
                line_info[ct] = (A_, B_, C_, cnt)
    return line_info


def _greedy_select(line_info, n_lines=27):
    """Greedy: pick top-n_lines by incidence count."""
    cand_list = sorted(line_info.items(), key=lambda kv: -kv[1][3])
    chosen = []
    total = 0
    for ct, (A_, B_, C_, cnt) in cand_list:
        if len(chosen) >= n_lines:
            break
        chosen.append(ct)
        total += cnt
    return chosen, total


def _solve_one(points, n_lines=27):
    """Solve: compute all lines, greedy select."""
    line_info = _compute_all_lines(points)
    if not line_info:
        return [], 0
    
    # Greedy initial selection
    chosen, total = _greedy_select(line_info, n_lines)
    
    # Build result
    chosen_coef = [line_info[ct][:3] for ct in chosen]
    return chosen_coef, total


def _build_points_grid(w, h, n_pts=54, rng=None):
    """Build point set from w x h grid, optionally removing random points."""
    if rng is None:
        rng = random
    all_pts = [(x, y) for x in range(w) for y in range(h)]
    if len(all_pts) <= n_pts:
        return all_pts
    return rng.sample(all_pts, n_pts)


def _smart_remove_points(all_pts, n_pts=54):
    """
    For a grid of size > n_pts, find the best n_pts points to keep
    by brute forcing the removal of (size - n_pts) points if the number
    of removals is small (<= 3).
    """
    size = len(all_pts)
    k_remove = size - n_pts
    
    if k_remove == 0:
        return all_pts
    
    if k_remove > 3:
        # Fallback to random if too many to remove
        rng = random.Random(12345)
        return rng.sample(all_pts, n_pts)
    
    # Precompute line incidences for the full grid to guide removal?
    # No, just brute force the subsets and evaluate.
    # This is expensive if k_remove is large, but for k_remove=2, it's ~1500 evaluations.
    
    best_score = -1
    best_subset = None
    
    # Get all indices
    indices = list(range(size))
    
    # Generate combinations of indices to REMOVE
    for remove_combo in combinations(indices, k_remove):
        remove_set = set(remove_combo)
        # Create the point set
        pts = [all_pts[i] for i in indices if i not in remove_set]
        
        # Evaluate
        lines, total = _solve_one(pts)
        if total > best_score:
            best_score = total
            best_subset = pts
            
        if _time_left() < 5:
            break
    
    if best_subset is not None:
        return best_subset
    return all_pts[:n_pts]


_best_score = 0
_best_points = None
_best_lines = None


def _finalize():
    """Validate and return best (points, lines) found, or fall back to seed."""
    global _best_points, _best_lines, _best_score
    if _best_points is not None and _best_lines is not None:
        pts = np.array(_best_points, dtype=np.int64)
        lns = np.array(_best_lines, dtype=np.int64)
        # Basic sanity checks
        if pts.shape == (54, 2) and lns.shape == (27, 3):
            # verify no A=B=0
            if not np.any((lns[:, 0] == 0) & (lns[:, 1] == 0)):
                return pts, lns
    # Fallback: seed construction
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


def _run_search():
    global _best_score, _best_points, _best_lines
    rng = random.Random(12345)
    
    # Seed baseline first
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    seed_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64).tolist()
    rows = [(a, -1, b) for a in range(1, k + 1) for b in range(1, k * k + 1)]
    seed_total = _evaluate(seed_pts, rows)
    _best_score = seed_total
    _best_points = [tuple(int(v) for v in p) for p in seed_pts]
    _best_lines = [tuple(int(v) for v in l) for l in rows]
    
    # Focus on grids where smart removal is feasible (size <= 57)
    configs = [
        (7, 8), (8, 7), # Size 56, remove 2
        (6, 9), (9, 6), # Size 54, full
        (5, 11), (11, 5), # Size 55, remove 1
        (10, 6), (6, 10), # Size 60, remove 6 (random)
        (12, 5), (5, 12), # Size 60, remove 6
        (4, 14), (14, 4), # Size 56, remove 2
        (3, 18), (18, 3), # Size 54, full
        (2, 27), (27, 2), # Size 54, full
        (1, 54), (54, 1), # Size 54, full
    ]
    
    for (w, h) in configs:
        if _time_left() < 30:
            break
        
        all_pts = [(x, y) for x in range(w) for y in range(h)]
        size = len(all_pts)
        
        if size == 54:
            points = all_pts
            lines, total = _solve_one(points)
            if total > _best_score:
                _best_score = total
                _best_points = [tuple(int(v) for v in p) for p in points]
                _best_lines = [tuple(int(v) for v in l) for l in lines]
        elif size > 54:
            if size - 54 <= 3:
                # Use smart removal
                points = _smart_remove_points(all_pts, 54)
                lines, total = _solve_one(points)
                if total > _best_score:
                    _best_score = total
                    _best_points = [tuple(int(v) for v in p) for p in points]
                    _best_lines = [tuple(int(v) for v in l) for l in lines]
            else:
                # Random removal, few trials
                for _ in range(5):
                    if _time_left() < 10: break
                    points = rng.sample(all_pts, 54)
                    lines, total = _solve_one(points)
                    if total > _best_score:
                        _best_score = total
                        _best_points = [tuple(int(v) for v in p) for p in points]
                        _best_lines = [tuple(int(v) for v in l) for l in lines]

def szemeredi_trotter_incidences():
    """
    Returns (points, lines): points a (54,2) int64 array, lines a (27,3) int64 array.
    Maximizes point-line incidences via greedy line selection over grid point sets with smart removal.
    """
    _run_search()
    return _finalize()


# EVOLVE-BLOCK-END
