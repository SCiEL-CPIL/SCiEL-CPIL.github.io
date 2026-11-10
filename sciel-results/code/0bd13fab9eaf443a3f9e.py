# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _compute_incidences(points, lines):
    """Compute total incidences."""
    total = 0
    for (A, B, C) in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += int(np.count_nonzero(vals == 0))
    return total


def _generate_candidate_lines(points, slopes):
    """Generate candidate lines for each slope family.
    For slope p/q (line: q*y - p*x + C = 0), find all distinct lines
    of that slope passing through at least one point.
    """
    x = points[:, 0].astype(np.int64)
    y = points[:, 1].astype(np.int64)
    line_set = set()
    
    for (p, q) in slopes:
        # Line: q*y - p*x + C = 0 => C = p*x - q*y
        Cs = p * x - q * y
        unique_cs = np.unique(Cs)
        for C_val in unique_cs:
            C_int = int(C_val)
            key = _normalize(-p, q, C_int)
            line_set.add(key)
    
    return line_set


def _select_best_lines(points, n_lines=27, slopes=None):
    """Select the best n_lines lines using greedy selection over mixed slopes."""
    if slopes is None:
        slopes = [(1,1), (-1,1), (2,1), (1,2), (-2,1), (-1,2),
                  (3,1), (1,3), (-3,1), (-1,3),
                  (0,1), (1,0)]
    
    # Generate candidate lines from slope families
    candidate_lines = _generate_candidate_lines(points, slopes)
    
    # Also add all pair-defined lines
    n = len(points)
    for i in range(n):
        xi, yi = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            xj, yj = int(points[j, 0]), int(points[j, 1])
            A = yi - yj
            B = xj - xi
            C = -(xi * yj - xj * yi)
            key = _normalize(A, B, C)
            candidate_lines.add(key)
    
    # Compute incidence count for each candidate line
    line_counts = {}
    for key in candidate_lines:
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        line_counts[key] = int(np.count_nonzero(vals == 0))
    
    # Sort by count descending, then by key for determinism
    candidates = sorted(line_counts.items(), key=lambda t: (-t[1], t[0]))
    
    selected = []
    selected_set = set()
    for key, cnt in candidates:
        selected.append(key)
        selected_set.add(key)
        if len(selected) == n_lines:
            break
    
    # If we don't have enough, that's fine (shouldn't happen with 54 points)
    return selected


def _try_grid(rows, cols, scale=1):
    """Try a rectangular grid of given dimensions with optional scaling."""
    xs = np.arange(1, cols + 1, dtype=np.int64) * scale
    ys = np.arange(1, rows + 1, dtype=np.int64) * scale
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) != 54:
        return None, None, 0
    
    lines_list = _select_best_lines(points, 27)
    if len(lines_list) < 27:
        return None, None, 0
    
    lines = np.array(lines_list, dtype=np.int64)
    incidences = _compute_incidences(points, lines)
    return points, lines, incidences


def _local_search(points, lines, max_iters=500, time_limit=300.0):
    """Local search: try moving individual points to nearby positions."""
    t_start = time.time()
    best_points = points.copy()
    best_lines = lines.copy()
    best_inc = _compute_incidences(best_points, best_lines)
    
    x_min = int(best_points[:, 0].min())
    x_max = int(best_points[:, 0].max())
    y_min = int(best_points[:, 1].min())
    y_max = int(best_points[:, 1].max())
    
    x_search_min = x_min - 2
    x_search_max = x_max + 2
    y_search_min = y_min - 2
    y_search_max = y_max + 2
    
    no_improve = 0
    
    for iteration in range(max_iters):
        if time.time() - t_start > time_limit:
            break
        if no_improve > 100:
            break
        
        improved = False
        for i in range(len(best_points)):
            if time.time() - t_start > time_limit:
                break
            orig_x, orig_y = int(best_points[i, 0]), int(best_points[i, 1])
            
            for dx in range(-2, 3):
                if improved:
                    break
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if new_x < x_search_min or new_x > x_search_max:
                        continue
                    if new_y < y_search_min or new_y > y_search_max:
                        continue
                    
                    # Check if position is already occupied
                    if np.any((best_points[:, 0] == new_x) & (best_points[:, 1] == new_y)):
                        continue
                    
                    test_points = best_points.copy()
                    test_points[i, 0] = new_x
                    test_points[i, 1] = new_y
                    
                    test_lines = _select_best_lines(test_points, 27)
                    if len(test_lines) < 27:
                        continue
                    test_lines_arr = np.array(test_lines, dtype=np.int64)
                    inc = _compute_incidences(test_points, test_lines_arr)
                    
                    if inc > best_inc:
                        best_points = test_points
                        best_lines = test_lines_arr
                        best_inc = inc
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_points, best_lines, best_inc


def szemeredi_trotter_incidences():
    t_start = time.time()
    deadline = 600.0
    margin = 60.0
    
    best_points = None
    best_lines = None
    best_inc = -1
    
    # Try various rectangular grids that multiply to 54
    grid_dims = [(1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)]
    
    for rows, cols in grid_dims:
        if time.time() > deadline - margin:
            break
        pts, lns, inc = _try_grid(rows, cols)
        if pts is not None and inc > best_inc:
            best_inc = inc
            best_points = pts.copy()
            best_lines = lns.copy()
    
    # Try scaled grids for the main shapes
    for scale in [2, 3]:
        if time.time() > deadline - margin:
            break
        for rows, cols in [(6, 9), (9, 6)]:
            pts, lns, inc = _try_grid(rows, cols, scale=scale)
            if pts is not None and inc > best_inc:
                best_inc = inc
                best_points = pts.copy()
                best_lines = lns.copy()
    
    # Try 7x8 grid with 2 points removed
    if time.time() <= deadline - margin:
        xs = np.arange(1, 9, dtype=np.int64)
        ys = np.arange(1, 8, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        full_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Try removing pairs of points (one from different rows)
        for rem1 in range(0, 56, 7):
            for rem2 in range(rem1 + 1, 56, 7):
                if time.time() > deadline - margin:
                    break
                mask = np.ones(56, dtype=bool)
                mask[rem1] = False
                mask[rem2] = False
                test_pts = full_pts[mask]
                if len(test_pts) != 54:
                    continue
                test_lines = _select_best_lines(test_pts, 27)
                if len(test_lines) < 27:
                    continue
                test_lines_arr = np.array(test_lines, dtype=np.int64)
                inc = _compute_incidences(test_pts, test_lines_arr)
                if inc > best_inc:
                    best_inc = inc
                    best_points = test_pts.copy()
                    best_lines = test_lines_arr
            if time.time() > deadline - margin:
                break
    
    # Local search on the best configuration
    if best_points is not None and best_lines is not None:
        remaining_time = deadline - margin - (time.time() - t_start)
        if remaining_time > 30:
            ls_points, ls_lines, ls_inc = _local_search(
                best_points, best_lines,
                max_iters=1000,
                time_limit=remaining_time
            )
            if ls_inc > best_inc:
                best_inc = ls_inc
                best_points = ls_points
                best_lines = ls_lines
    
    # Fallback
    if best_points is None or best_lines is None:
        pts, lns, _ = _try_grid(9, 6)
        best_points = pts
        best_lines = lns
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
