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
    
    line_points = {}
    
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
            
            if canon not in line_points:
                line_points[canon] = set()
            
            line_points[canon].add(i)
            line_points[canon].add(j)
            
    # Now count the actual number of points on each line
    final_counts = []
    for canon, idx_set in line_points.items():
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
    Try a specific grid size. If it has more than num_points, we just take a subset?
    No, the task requires exactly 54 points.
    So we only consider grids with exactly 54 points or subsets.
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


def _local_search(points, lines, max_iters=50, time_limit=60.0):
    """
    Local search: try swapping lines and moving points to improve incidence count.
    """
    start_time = time.time()
    n = len(points)
    m = len(lines)
    
    points = points.copy()
    lines = lines.copy()
    current_score = _count_incidences(points, lines)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        if time.time() - start_time > time_limit:
            break
        
        # Try moving each point to a nearby position
        for pi in range(n):
            if time.time() - start_time > time_limit:
                break
            
            x, y = points[pi]
            current_inc = np.sum((lines[:, 0] * x + lines[:, 1] * y + lines[:, 2]) == 0)
            
            best_gain = 0
            best_pos = None
            # Try nearby positions in a 5x5 neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    # Check if this position is already occupied
                    occupied = False
                    for pj in range(n):
                        if pj != pi and points[pj, 0] == nx and points[pj, 1] == ny:
                            occupied = True
                            break
                    if occupied:
                        continue
                    new_inc = np.sum((lines[:, 0] * nx + lines[:, 1] * ny + lines[:, 2]) == 0)
                    gain = new_inc - current_inc
                    if gain > best_gain:
                        best_gain = gain
                        best_pos = (nx, ny)
            
            if best_pos is not None and best_gain > 0:
                points[pi] = best_pos
                current_score += best_gain
                improved = True
        
        # Try swapping lines
        for li in range(m):
            if time.time() - start_time > time_limit:
                break
            
            a, b, c = lines[li]
            current_inc_line = np.sum((points[:, 0] * a + points[:, 1] * b + c) == 0)
            
            # Try to find a better line through some pair of points
            best_gain = 0
            best_line = None
            # Sample some pairs to keep it fast
            for _ in range(50):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                na = y1 - y2
                nb = x2 - x1
                nc = x1 * y2 - x2 * y1
                canon = _canonical_line(na, nb, nc)
                
                # Check if this line is already in our set
                already_there = False
                for lj in range(m):
                    if lj == li:
                        continue
                    ca, cb, cc = lines[lj]
                    canon_j = _canonical_line(int(ca), int(cb), int(cc))
                    if canon == canon_j:
                        already_there = True
                        break
                if already_there:
                    continue
                
                new_inc_line = np.sum((points[:, 0] * canon[0] + points[:, 1] * canon[1] + canon[2]) == 0)
                gain = new_inc_line - current_inc_line
                if gain > best_gain:
                    best_gain = gain
                    best_line = canon
            
            if best_line is not None and best_gain > 0:
                lines[li] = best_line
                current_score += best_gain
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

    # Local search to improve
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 30:
        search_time = min(60.0, DEADLINE - SAFETY_MARGIN - 30 - (time.time() - start_time))
        if search_time > 10:
            pts, lns, score = _local_search(best_points, best_lines, max_iters=30, time_limit=search_time)
            if score > best_score:
                best_score = score
                best_points = pts
                best_lines = lns

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
