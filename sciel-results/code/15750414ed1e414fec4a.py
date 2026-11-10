# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time


def _normalize_line(A, B, C):
    """Normalize a line so the first nonzero of (A,B,C) is positive and gcd is 1."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count incidences: number of (point, line) pairs where A*x+B*y+C==0."""
    # points: (n,2) int64, lines: (m,3) int64
    # For each line, compute A*x + B*y + C for all points
    total = 0
    for i in range(lines.shape[0]):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total


def _greedy_select_lines(points, n_lines, deadline):
    """
    Greedily select n_lines lines to maximize incidences.
    Start with all grid lines (vertical, horizontal, main diagonals),
    then greedily add the line through any pair of points that maximizes
    marginal incidence gain.
    
    Returns (lines_array, total_incidences).
    """
    n_pts = points.shape[0]
    
    # Build a set of all unique lines through pairs of points
    # Plus all "obvious" grid lines
    line_to_points = {}  # normalized line -> set of point indices on it
    
    # Precompute all lines through pairs
    line_set = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line through (x1,y1) and (x2,y2): dy*(x-x1) - dx*(y-y1) = 0
            # => dy*x - dx*y + (dx*y1 - dy*x1) = 0
            A = int(dy)
            B = int(-dx)
            C = int(dx * y1 - dy * x1)
            norm = _normalize_line(A, B, C)
            line_set.add(norm)
    
    # Also add axis-aligned lines through each point (in case they're not in pair set)
    for i in range(n_pts):
        x, y = int(points[i, 0]), int(points[i, 1])
        # Vertical: x = const
        norm_v = _normalize_line(1, 0, -x)
        line_set.add(norm_v)
        # Horizontal: y = const
        norm_h = _normalize_line(0, 1, -y)
        line_set.add(norm_h)
    
    # For each candidate line, compute which points lie on it
    candidate_lines = list(line_set)
    line_point_masks = {}
    for ln in candidate_lines:
        A, B, C = ln
        vals = A * points[:, 0] + B * points[:, 1] + C
        mask = (vals == 0)
        line_point_masks[ln] = mask
    
    # Greedy selection
    selected = []
    selected_set = set()
    current_mask = np.zeros(n_pts, dtype=bool)  # points already covered
    total_inc = 0
    
    for _ in range(n_lines):
        if time.time() > deadline:
            break
        best_gain = -1
        best_line = None
        for ln in candidate_lines:
            if ln in selected_set:
                continue
            mask = line_point_masks[ln]
            # Marginal gain: points on this line not yet "counted" by any selected line
            # Actually, we want to maximize total incidences, so each line contributes
            # its full count regardless of overlap. So gain = count of points on line.
            gain = np.sum(mask)
            if gain > best_gain:
                best_gain = gain
                best_line = ln
        if best_line is None:
            break
        selected.append(best_line)
        selected_set.add(best_line)
        total_inc += best_gain
    
    lines_array = np.array(selected, dtype=np.int64)
    return lines_array, total_inc


def _try_grid(w, h, n_lines, deadline):
    """Try a w x h grid and return (points, lines, incidences)."""
    xs = np.arange(1, w + 1)
    ys = np.arange(1, h + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines, inc = _greedy_select_lines(points, n_lines, deadline)
    return points, lines, inc


def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Tries multiple grid configurations (6x9, 5x11, 4x14, 3x18, 2x27, 9x6, 11x5, 14x4, 18x3, 27x2)
    and selects the one with maximum incidences using greedy line selection.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit
    
    best_points = None
    best_lines = None
    best_inc = -1
    
    # Try various grid dimensions that multiply to 54
    grid_dims = [(6, 9), (9, 6), (5, 11), (11, 5), (4, 14), (14, 4),
                 (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)]
    
    for w, h in grid_dims:
        if time.time() > deadline - 10:
            break
        try:
            pts, lns, inc = _try_grid(w, h, 27, deadline)
            if inc > best_inc:
                best_inc = inc
                best_points = pts
                best_lines = lns
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: 6x9 grid with hand-picked lines
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        for b in [-1, 0, 1, 2, 3, 4]:
            lines.append((1, -1, b))
        for b in [6, 7, 8, 9, 10, 11]:
            lines.append((1, 1, -b))
        best_lines = np.array(lines, dtype=np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
