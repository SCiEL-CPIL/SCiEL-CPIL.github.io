import numpy as np
import time
from math import gcd


def _gcd3(a, b, c):
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g if g != 0 else 0


def _normalize_line(A, B, C):
    """Normalize line coefficients: divide by gcd, make first nonzero positive."""
    if A == 0 and B == 0:
        return None
    g = _gcd3(A, B, C)
    if g != 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _incidence_count(points, lines):
    """Count (p,l) pairs where A*x + B*y + C == 0."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    val = A * x + B * y + C
    return int(np.count_nonzero(val == 0))


def _line_incidence(A, B, C, points):
    """Count incidences for a single line against all points."""
    val = A * points[:, 0] + B * points[:, 1] + C
    return int(np.count_nonzero(val == 0))


def _build_initial(points):
    """
    Build initial set of 27 lines for given 54 points.
    Generate all lines through pairs of points, score them, pick top 27.
    """
    n_pts = len(points)
    line_set = set()

    # Add axis-aligned lines
    xs = points[:, 0]
    ys = points[:, 1]
    for x_val in np.unique(xs):
        norm = _normalize_line(1, 0, -int(x_val))
        if norm:
            line_set.add(norm)
    for y_val in np.unique(ys):
        norm = _normalize_line(0, 1, -int(y_val))
        if norm:
            line_set.add(norm)

    # Generate lines through all pairs of points
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            A = int(y2 - y1)
            B = int(x1 - x2)
            C = int(x2 * y1 - x1 * y2)
            norm = _normalize_line(A, B, C)
            if norm:
                line_set.add(norm)

    # Score each line
    lines_list = list(line_set)
    scores = np.array([_line_incidence(A, B, C, points) for A, B, C in lines_list])

    # Pick top 27
    top_idx = np.argsort(-scores)[:27]
    selected = [lines_list[i] for i in top_idx]
    lines = np.array(selected, dtype=np.int64)
    return lines, scores, lines_list


def _local_search(points, lines, max_time=300):
    """
    Local search: try swapping each line with a better candidate.
    Returns improved (points, lines, count).
    """
    start_time = time.time()
    n_lines = len(lines)
    n_pts = len(points)

    # Precompute per-line incidence
    line_scores = np.array([
        _line_incidence(lines[i, 0], lines[i, 1], lines[i, 2], points)
        for i in range(n_lines)
    ])
    total = int(line_scores.sum())

    # Build candidate pool: all lines through pairs of points
    line_set = set()
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            A = int(y2 - y1)
            B = int(x1 - x2)
            C = int(x2 * y1 - x1 * y2)
            norm = _normalize_line(A, B, C)
            if norm:
                line_set.add(norm)

    # Also add axis-aligned
    for x_val in np.unique(points[:, 0]):
        norm = _normalize_line(1, 0, -int(x_val))
        if norm:
            line_set.add(norm)
    for y_val in np.unique(points[:, 1]):
        norm = _normalize_line(0, 1, -int(y_val))
        if norm:
            line_set.add(norm)

    candidates = list(line_set)
    # Precompute candidate scores
    cand_scores = np.array([
        _line_incidence(c[0], c[1], c[2], points) for c in candidates
    ])

    # Map from normalized tuple to index in candidates
    cand_map = {c: i for i, c in enumerate(candidates)}

    # Current line set as normalized tuples
    current_lines = [tuple(lines[i]) for i in range(n_lines)]
    current_set = set(current_lines)

    improved = True
    while improved:
        improved = False
        for i in range(n_lines):
            if time.time() - start_time > max_time:
                break
            # Find best candidate not in current set
            best_gain = 0
            best_cand = None
            for ci, c in enumerate(candidates):
                if c in current_set:
                    continue
                gain = cand_scores[ci] - line_scores[i]
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
            if best_cand is not None and best_gain > 0:
                # Swap
                current_set.discard(current_lines[i])
                current_lines[i] = best_cand
                current_set.add(best_cand)
                line_scores[i] = cand_scores[candidates.index(best_cand)]
                total += best_gain
                improved = True
        if time.time() - start_time > max_time:
            break

    new_lines = np.array(current_lines, dtype=np.int64)
    return new_lines, total


def _try_grid(cols, rows):
    """Build a cols x rows grid and find best 27 lines."""
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)

    if len(points) != 54:
        return None, None, 0

    lines, scores, lines_list = _build_initial(points)
    inc = _incidence_count(points, lines)
    return points, lines, inc


def _baseline():
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


def szemeredi_trotter_incidences():
    deadline = time.time() + 620

    best_points, best_lines, best_inc = None, None, 0

    # Try different grid configurations
    grids = [(9, 6), (6, 9), (18, 3), (3, 18), (27, 2), (2, 27), (1, 54), (54, 1)]
    for cols, rows in grids:
        if cols * rows != 54:
            continue
        pts, lns, inc = _try_grid(cols, rows)
        if pts is not None and inc > best_inc:
            best_points, best_lines, best_inc = pts, lns, inc

    if best_points is None:
        return _baseline()

    # Local search on the best grid
    remaining = deadline - time.time()
    if remaining > 30:
        ls_time = min(remaining - 10, 300)
        new_lines, new_inc = _local_search(best_points, best_lines, max_time=ls_time)
        if new_inc > best_inc:
            best_lines, best_inc = new_lines, new_inc

    return best_points, best_lines
