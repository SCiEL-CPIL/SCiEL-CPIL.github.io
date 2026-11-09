# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    """Get normalized line (A,B,C) through two integer points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _count_incidence(points, line):
    """Count how many points lie on the line."""
    A, B, C = line
    count = 0
    for x, y in points:
        if A * x + B * y + C == 0:
            count += 1
    return count

def _build_all_lines(points):
    """Build all unique lines determined by pairs of points, with incidence counts."""
    n = len(points)
    line_to_idx = {}
    lines = []  # list of (A, B, C)
    incidences = []  # list of counts
    
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_two_points(points[i], points[j])
            if line not in line_to_idx:
                line_to_idx[line] = len(lines)
                lines.append(line)
                incidences.append(0)
            idx = line_to_idx[line]
            # Count incidences for this line
            A, B, C = line
            cnt = 0
            for k in range(n):
                x, y = points[k]
                if A * x + B * y + C == 0:
                    cnt += 1
            incidences[idx] = cnt
    
    return lines, incidences

def _select_best_lines(lines, incidences, m):
    """Select m lines that maximize total incidence (greedy with conflict handling)."""
    n_lines = len(lines)
    if m >= n_lines:
        return list(range(n_lines))
    
    # Greedy: pick lines in order of incidence count, but avoid duplicates
    # Since lines are already unique, just pick top m by incidence
    order = sorted(range(n_lines), key=lambda i: -incidences[i])
    selected = order[:m]
    return selected

def _try_grid(rows, cols, deadline):
    """Try a grid of given dimensions and return best score."""
    points_list = []
    for x in range(1, cols + 1):
        for y in range(1, rows + 1):
            points_list.append((x, y))
    
    n_pts = len(points_list)
    if n_pts != 54:
        return 0, None, None
    
    # Build all lines
    lines, incidences = _build_all_lines(points_list)
    
    if time.time() > deadline:
        return 0, None, None
    
    # Select best 27 lines
    selected = _select_best_lines(lines, incidences, 27)
    
    total = sum(incidences[i] for i in selected)
    
    # Build output arrays
    points_arr = np.array(points_list, dtype=np.int64)
    lines_arr = np.array([lines[i] for i in selected], dtype=np.int64)
    
    return total, points_arr, lines_arr

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct 54 points and 27 lines
    to maximize point-line incidences.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_score = 0
    best_points = None
    best_lines = None
    
    # Try various grid shapes that give 54 points
    # 54 = 1*54, 2*27, 3*18, 6*9, 9*6, 18*3, 27*2, 54*1
    grid_shapes = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2),
        (1, 54), (54, 1),
    ]
    
    for rows, cols in grid_shapes:
        if time.time() > deadline:
            break
        score, pts, lns = _try_grid(rows, cols, deadline)
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
    
    # If we didn't find anything, fall back to the original construction
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
