# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import product


def _count_incidentences(points, lines):
    """Count (point, line) pairs where A*x + B*y + C == 0 exactly."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _normalize_line(A, B, C):
    """Normalize (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_key(A, B, C):
    return _normalize_line(A, B, C)


def _is_valid(points, lines):
    """Check all validity constraints."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check points: no NaN/inf, integer-valued, no repeats
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    # Check lines: no A=B=0, no repeats after normalization
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True


def _build_grid_with_diagonals():
    """Build a 6x9 grid with vertical, horizontal, and diagonal lines."""
    # Points: 6 columns x 9 rows
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    points = [(x, y) for x in xs for y in ys]
    points = np.array(points, dtype=np.int64)

    lines = []
    seen = set()

    def add_line(A, B, C):
        key = _line_key(A, B, C)
        if key not in seen and not (key[0] == 0 and key[1] == 0):
            seen.add(key)
            lines.append(key)

    # Vertical lines: x = c -> (1, 0, -c)
    for c in xs:
        add_line(1, 0, -c)

    # Horizontal lines: y = r -> (0, 1, -r)
    for r in ys:
        add_line(0, 1, -r)

    # Diagonals with slope 1: y - x = d -> (1, -1, d)
    # y - x = d, for d in range. In 6x9 grid, d ranges from 1-6=-5 to 9-1=8
    for d in range(-5, 9):
        add_line(1, -1, d)

    # Diagonals with slope -1: y + x = s -> (1, 1, -s)
    # x + y = s, s ranges from 2 to 15
    for s in range(2, 16):
        add_line(1, 1, -s)

    # We have 6 + 9 + 14 + 14 = 43 candidate lines. Pick top 27 by incidence count.
    # Compute incidences for each candidate
    point_set = set((int(p[0]), int(p[1])) for p in points)
    cand_inc = []
    for (A, B, C) in lines:
        cnt = 0
        for (x, y) in point_set:
            if A * x + B * y + C == 0:
                cnt += 1
        cand_inc.append((cnt, A, B, C))

    cand_inc.sort(reverse=True, key=lambda t: t[0])
    top27 = [(A, B, C) for (_, A, B, C) in cand_inc[:27]]
    lines_arr = np.array(top27, dtype=np.int64)

    return points, lines_arr


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    Uses a grid + diagonal construction, selecting the 27 lines with highest incidence.
    """
    deadline = time.time() + 600
    safety_margin = 30  # seconds

    # Start with the grid + diagonals construction
    points, lines = _build_grid_with_diagonals()

    if not _is_valid(points, lines):
        # Fallback to known valid construction
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

    best_points, best_lines = points, lines
    best_count = _count_incidentences(points, lines)

    # Try to improve with local search if time permits
    if time.time() < deadline - safety_margin:
        # Simple local search: try replacing one line at a time
        for _ in range(200):  # limited iterations
            if time.time() > deadline - safety_margin:
                break
            # Try adding a random line and removing the worst
            # Generate candidate lines through pairs of points
            n = len(points)
            best_replace = None
            best_gain = 0
            # Sample some candidate lines
            for _trial in range(50):
                i1 = np.random.randint(n)
                i2 = np.random.randint(n)
                if i1 == i2:
                    continue
                x1, y1 = points[i1]
                x2, y2 = points[i2]
                # Line through (x1,y1) and (x2,y2)
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                if A == 0 and B == 0:
                    continue
                key = _line_key(A, B, C)
                if key[0] == 0 and key[1] == 0:
                    continue
                # Check if already in lines
                line_keys = [_line_key(int(l[0]), int(l[1]), int(l[2])) for l in lines]
                if key in line_keys:
                    continue
                # Try replacing each existing line
                for j in range(len(lines)):
                    new_lines = list(lines)
                    new_lines[j] = np.array(key, dtype=np.int64)
                    new_lines_arr = np.array(new_lines, dtype=np.int64)
                    if not _is_valid(points, new_lines_arr):
                        continue
                    cnt = _count_incidentences(points, new_lines_arr)
                    gain = cnt - best_count
                    if gain > best_gain:
                        best_gain = gain
                        best_replace = (j, key, cnt)
            if best_replace is not None:
                j, key, new_cnt = best_replace
                lines[j] = np.array(key, dtype=np.int64)
                best_count = new_cnt
                best_points, best_lines = points, lines

    return best_points, best_lines


# EVOLVE-BLOCK-END
