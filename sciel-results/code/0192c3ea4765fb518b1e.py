import numpy as np
import time
import math
from itertools import combinations
from collections import Counter

DEADLINE = 630.0
SAFETY_MARGIN = 10.0


def _canonical_line(a, b, c):
    """Return canonical form of line (a,b,c): divide by gcd, make first nonzero positive."""
    if a == 0 and b == 0 and c == 0:
        return (0, 0, 0)
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    a //= g
    b //= g
    c //= g
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0 and b < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)


def _count_incidences(points, lines):
    """Count point-line incidences exactly using numpy."""
    if points.size == 0 or lines.size == 0:
        return 0
    vals = points @ lines[:, :2].T + lines[:, 2:3].T
    count = np.sum(vals == 0)
    return int(count)


def _generate_grid(w, h):
    """Generate a w x h grid of points."""
    xs = np.arange(w)
    ys = np.arange(h)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _generate_sheared_grid(w, h, shear=1):
    """Generate a sheared w x h grid: points (i, j) -> (i + shear*j, j)."""
    xs = np.arange(w)
    ys = np.arange(h)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    sx = px + shear * py
    points = np.stack([sx.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _find_best_lines(points, num_lines):
    """
    Find the top num_lines lines (by number of point incidences) from all pair-defined lines.
    """
    n = len(points)
    if n < 2:
        return []
    
    line_counts = {}
    
    # Generate all lines through pairs of points
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line through (x1,y1) and (x2,y2): (y1-y2)x + (x2-x1)y + (x1*y2 - x2*y1) = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            canon = _canonical_line(a, b, c)
            
            if canon not in line_counts:
                line_counts[canon] = set()
            
            line_counts[canon].add(i)
            line_counts[canon].add(j)
            
    # Now count the actual number of points on each line
    final_counts = []
    for canon, idx_set in line_counts.items():
        a, b, c = canon
        cnt = 0
        for p_idx in range(n):
            x, y = points[p_idx]
            if a * x + b * y + c == 0:
                cnt += 1
        final_counts.append((cnt, canon))
        
    # Sort by count descending
    final_counts.sort(key=lambda x: -x[0])
    top = [lc[1] for lc in final_counts[:num_lines]]
    return top


def _evaluate(points, lines):
    """
    Evaluate the configuration. Returns the number of incidences.
    Also performs validity checks.
    """
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return -1
    
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return -1
    
    if not np.allclose(points, np.round(points), atol=1e-6):
        return -1
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return -1
    
    points_int = np.round(points).astype(np.int64)
    lines_int = np.round(lines).astype(np.int64)
    
    if len(set(map(tuple, points_int.tolist()))) != 54:
        return -1
    
    seen_lines = set()
    for i in range(27):
        a, b, c = lines_int[i]
        if a == 0 and b == 0:
            return -1
        canon = _canonical_line(int(a), int(b), int(c))
        if canon in seen_lines:
            return -1
        seen_lines.add(canon)
    
    count = _count_incidences(points_int, lines_int)
    return count


def _try_grid(w, h, num_points=54, num_lines=27):
    """
    Try a specific grid size.
    """
    if w * h != num_points:
        return None, None, 0
    
    points = _generate_grid(w, h)
    top_lines = _find_best_lines(points, num_lines)
    
    if len(top_lines) < num_lines:
        return None, None, 0
        
    lines = np.array(top_lines, dtype=np.int64)
    score = _count_incidences(points, lines)
    
    return points, lines, score


def _try_sheared_grid(w, h, shear, num_points=54, num_lines=27):
    """
    Try a sheared grid.
    """
    if w * h != num_points:
        return None, None, 0
    
    points = _generate_sheared_grid(w, h, shear)
    top_lines = _find_best_lines(points, num_lines)
    
    if len(top_lines) < num_lines:
        return None, None, 0
        
    lines = np.array(top_lines, dtype=np.int64)
    score = _count_incidences(points, lines)
    
    return points, lines, score


def _local_search(points, lines, max_iter=500, time_limit=300.0):
    """
    Local search: try replacing individual points with nearby lattice points
    to increase total incidences.
    """
    start_time = time.time()
    n = len(points)
    m = len(lines)
    
    current_score = _count_incidences(points, lines)
    
    # Precompute: for each point, which lines it's on
    point_on_lines = []
    for i in range(n):
        x, y = points[i]
        on = []
        for j in range(m):
            a, b, c = lines[j]
            if a * x + b * y + c == 0:
                on.append(j)
        point_on_lines.append(on)
    
    # For each line, which points are on it
    line_on_points = []
    for j in range(m):
        a, b, c = lines[j]
        on = []
        for i in range(n):
            x, y = points[i]
            if a * x + b * y + c == 0:
                on.append(i)
        line_on_points.append(on)
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if time.time() - start_time > time_limit:
            break
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                    break
            
            old_x, old_y = points[i]
            # Lines that point i is currently on
            old_lines = point_on_lines[i]
            old_inc = len(old_lines)
            
            # Try moving to nearby lattice points
            best_delta = 0
            best_new_pt = None
            
            # Search radius: try a few steps in each direction
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    # Check if this point is already used
                    if any(points[k][0] == new_x and points[k][1] == new_y for k in range(n) if k != i):
                        continue
                    
                    # Count new incidences
                    new_inc = 0
                    for j in range(m):
                        a, b, c = lines[j]
                        if a * new_x + b * new_y + c == 0:
                            new_inc += 1
                    
                    delta = new_inc - old_inc
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pt = (new_x, new_y)
            
            if best_new_pt is not None and best_delta > 0:
                # Apply the move
                points[i] = best_new_pt
                new_x, new_y = best_new_pt
                
                # Update point_on_lines
                point_on_lines[i] = [j for j in range(m) if lines[j][0]*new_x + lines[j][1]*new_y + lines[j][2] == 0]
                
                # Update line_on_points
                for j in range(m):
                    a, b, c = lines[j]
                    if a*new_x + b*new_y + c == 0:
                        if i not in line_on_points[j]:
                            line_on_points[j].append(i)
                    else:
                        if i in line_on_points[j]:
                            line_on_points[j].remove(i)
                
                current_score += best_delta
                improved = True
    
    return points, lines, current_score


def _local_search_with_line_update(points, lines, max_iter=100, time_limit=120.0):
    """
    Enhanced local search: after moving a point, re-select best lines.
    More expensive but can find better solutions.
    """
    start_time = time.time()
    n = len(points)
    m = len(lines)
    
    current_score = _count_incidences(points, lines)
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if time.time() - start_time > time_limit:
            break
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            old_x, old_y = points[i]
            
            # Try moving to nearby lattice points
            best_delta = 0
            best_new_pt = None
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    # Check if this point is already used
                    if any(points[k][0] == new_x and points[k][1] == new_y for k in range(n) if k != i):
                        continue
                    
                    # Temporarily move the point
                    points[i] = [new_x, new_y]
                    
                    # Re-find best lines
                    new_lines = _find_best_lines(points, m)
                    if len(new_lines) < m:
                        points[i] = [old_x, old_y]
                        continue
                    
                    new_lines_arr = np.array(new_lines, dtype=np.int64)
                    new_score = _count_incidences(points, new_lines_arr)
                    delta = new_score - current_score
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pt = (new_x, new_y)
                        best_new_lines = new_lines_arr
                        best_new_score = new_score
                    
                    # Restore
                    points[i] = [old_x, old_y]
            
            if best_new_pt is not None and best_delta > 0:
                points[i] = best_new_pt
                lines = best_new_lines
                current_score = best_new_score
                improved = True
    
    return points, lines, current_score


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.
    """
    start_time = time.time()
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Try exact grids
    grid_configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
    ]
    
    for w, h in grid_configs:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        pts, lns, score = _try_grid(w, h, 54, 27)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lns.copy()
    
    # Try sheared grids
    shear_configs = [
        (6, 9, 1), (9, 6, 1), (3, 18, 1), (18, 3, 1),
        (6, 9, -1), (9, 6, -1), (3, 18, -1), (18, 3, -1),
        (6, 9, 2), (9, 6, 2), (3, 18, 2),
        (6, 9, -2), (9, 6, -2),
    ]
    
    for w, h, shear in shear_configs:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        pts, lns, score = _try_sheared_grid(w, h, shear, 54, 27)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lns.copy()
    
    # If no grid worked (shouldn't happen), fallback to a simple construction
    if best_points is None:
        # Fallback: 54 points on x-axis, 27 vertical lines
        points = np.zeros((54, 2), dtype=np.int64)
        points[:, 0] = np.arange(54)
        lines = np.zeros((27, 3), dtype=np.int64)
        for i in range(27):
            lines[i] = [1, 0, -i]
        best_score = _count_incidences(points, lines)
        best_points = points
        best_lines = lines
    
    # Local search: try moving individual points
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 60:
        ls_time_limit = min(120.0, DEADLINE - SAFETY_MARGIN - (time.time() - start_time) - 30)
        if ls_time_limit > 10:
            ls_points, ls_lines, ls_score = _local_search(best_points, best_lines, max_iter=200, time_limit=ls_time_limit)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points
                best_lines = ls_lines
    
    # Enhanced local search with line re-optimization (if time permits)
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 30:
        ls_time_limit = min(60.0, DEADLINE - SAFETY_MARGIN - (time.time() - start_time) - 15)
        if ls_time_limit > 10:
            ls_points, ls_lines, ls_score = _local_search_with_line_update(best_points, best_lines, max_iter=50, time_limit=ls_time_limit)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points
                best_lines = ls_lines
    
    # Final validation
    final_score = _evaluate(best_points, best_lines)
    if final_score < 0:
        # Fallback to a known valid configuration
        # 3x18 grid
        pts, lns, _ = _try_grid(3, 18, 54, 27)
        if pts is not None:
            return pts, lns
        else:
            # Hardcoded fallback
            points = np.zeros((54, 2), dtype=np.int64)
            points[:, 0] = np.arange(54)
            lines = np.zeros((27, 3), dtype=np.int64)
            for i in range(27):
                lines[i] = [1, 0, -i]
            return points, lines

    return best_points, best_lines
