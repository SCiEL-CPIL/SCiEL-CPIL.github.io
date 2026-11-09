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
            
            if canon in line_counts:
                # We don't need to recount if we already have the line, 
                # but we need the count. 
                # Efficient way: just add to a set of points on the line? 
                # Or just recount at the end.
                pass
            else:
                line_counts[canon] = set()
                
            # Add points i and j to the set for this line
            line_counts[canon].add(i)
            line_counts[canon].add(j)
            
    # Now count the actual number of points on each line
    # This is O(N^2 * L) which might be slow for large N, but N=54 is small.
    # L is number of lines ~ N^2/2 = 1431.
    # 1431 * 54 ~ 77k operations. Very fast.
    
    final_counts = []
    for canon, idx_set in line_counts.items():
        a, b, c = canon
        # Verify count by checking all points
        # This ensures accuracy
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
    Try a specific grid size. If it has more than num_points, we just take a subset?
    No, the task requires exactly 54 points.
    So we only consider grids with exactly 54 points or subsets.
    Subsets are harder to optimize. 
    However, 54 = 6*9 = 3*18 = 2*27 = 1*54.
    Let's try these exact grids first.
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
            best_points = pts
            best_lines = lns
            
    # If no exact grid worked (shouldn't happen), fallback to a simple construction
    if best_points is None:
        # Fallback: 54 points on x-axis, 27 vertical lines
        points = np.zeros((54, 2), dtype=np.int64)
        points[:, 0] = np.arange(54)
        lines = np.zeros((27, 3), dtype=np.int64)
        for i in range(27):
            lines[i] = [1, 0, -i] # x - i = 0
        best_score = _count_incidences(points, lines)
        best_points = points
        best_lines = lines

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
