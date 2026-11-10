# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd


def _gcd3(a, b, c):
    g = gcd(abs(int(a)), abs(int(b)))
    g = gcd(g, abs(int(c)))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    A //= g
    B //= g
    C //= g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidents(points, lines):
    vals = lines[:, 0][None, :] * points[:, 0:1] + lines[:, 1][None, :] * points[:, 1:2] + lines[:, 2][None, :]
    return int(np.sum(vals == 0))


def _best_lines_for_points(points, num_lines=27, max_time=30.0):
    """
    Given fixed points, find the best num_lines distinct lines.
    A line is a candidate if it passes through at least 2 points (or we allow any line).
    Actually, any line is a candidate, but lines with more points are better.
    We generate all lines through pairs of points.
    """
    n = points.shape[0]
    deadline = time.time() + max_time
    
    # Generate all unique lines determined by pairs of points
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            nl = _normalize_line(A, B, C)
            line_set.add(nl)
    
    # Score each line
    scored = []
    for nl in line_set:
        A, B, C = nl
        vals = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(vals == 0))
        scored.append((cnt, nl))
    
    # Sort by count desc
    scored.sort(key=lambda x: -x[0])
    
    selected = []
    selected_set = set()
    for cnt, nl in scored:
        if nl in selected_set:
            continue
        selected.append(nl)
        selected_set.add(nl)
        if len(selected) == num_lines:
            break
    
    # If we didn't find enough (unlikely for 54 points, which give ~1400 pairs -> many unique lines)
    # Pad with arbitrary lines if needed (e.g. horizontal lines with no points)
    if len(selected) < num_lines:
        # Try to add lines that pass through at least 1 point? No, we need distinct lines.
        # We can just add lines with 0 points, but that's bad.
        # Better: check if we can add lines passing through 1 point but not 2?
        # Actually, with 54 points, the number of distinct lines through pairs is huge.
        # 54*53/2 = 1431 pairs. Even if many are parallel, there are plenty.
        # Let's just pad with horizontal lines y=c for c not in point y-coords.
        y_coords = set(int(p[1]) for p in points)
        c = -100
        while len(selected) < num_lines:
            nl = _normalize_line(0, 1, c)
            if nl not in selected_set:
                selected.append(nl)
                selected_set.add(nl)
            c += 1
        
    lines_arr = np.array(selected, dtype=np.int64)
    return lines_arr


def _local_search_points(points, lines, max_time=60.0):
    """
    Local search: move one point at a time to a better position.
    """
    best_points = points.copy()
    best_count = _count_incidents(best_points, lines)
    deadline = time.time() + max_time
    n = 54
    m = 27
    
    A_arr = lines[:, 0]
    B_arr = lines[:, 1]
    C_arr = lines[:, 2]
    
    # Candidate positions: near existing points
    # We can generate candidates based on lines.
    # If we move point i to (x,y), it's good if it lies on many lines.
    # Let's just try integer coordinates in a box around the current point.
    
    no_improve = 0
    max_no_improve = 50
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 2 and no_improve < max_no_improve:
        improved = False
        # Shuffle point order
        order = np.arange(n)
        rng.shuffle(order)
        
        for i in order:
            if time.time() > deadline - 2:
                break
            
            orig_x, orig_y = int(best_points[i, 0]), int(best_points[i, 1])
            
            # Try a few random nearby positions
            # Also try positions on existing lines?
            # Let's just try a box of size 5x5 around current position
            # And also try positions on lines that currently pass through the point
            # Actually, moving a point *off* a line reduces count. Moving it *onto* a line increases.
            # So we want to find (x,y) that lies on many lines.
            
            # Heuristic: try intersections of pairs of lines?
            # There are 27 lines -> 351 intersections.
            # We can precompute these.
            
            # For now, let's just do random local moves.
            best_delta = 0
            best_pos = None
            
            # Try random moves
            for _ in range(20):
                dx = rng.integers(-5, 6)
                dy = rng.integers(-5, 6)
                nx, ny = orig_x + dx, orig_y + dy
                
                # Check collision
                if any(best_points[j, 0] == nx and best_points[j, 1] == ny for j in range(n) if j != i):
                    continue
                
                # Count incidences for new pos
                new_on = np.sum(A_arr * nx + B_arr * ny + C_arr == 0)
                old_on = np.sum(A_arr * orig_x + B_arr * orig_y + C_arr == 0)
                
                delta = new_on - old_on
                if delta > best_delta:
                    best_delta = delta
                    best_pos = (nx, ny)
            
            if best_pos is not None and best_delta > 0:
                best_points[i, 0] = best_pos[0]
                best_points[i, 1] = best_pos[1]
                best_count += best_delta
                improved = True
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            
    return best_points, best_count


def _try_grid_configuration(rows, cols, start_time, deadline):
    """
    Try a grid of size rows x cols.
    If rows*cols > 54, we need to remove points.
    If rows*cols < 54, we can't use it (need exactly 54).
    """
    if rows * cols != 54:
        return None, 0
        
    points_x = np.arange(rows)
    points_y = np.arange(cols)
    px, py = np.meshgrid(points_x, points_y, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = _best_lines_for_points(points, 27, max_time=10.0)
    count = _count_incidents(points, lines)
    
    # Local search
    if time.time() < deadline - 10:
        pts_ls, count_ls = _local_search_points(points, lines, max_time=min(30.0, deadline - time.time() - 5))
        if count_ls > count:
            points = pts_ls
            count = count_ls
            # Re-optimize lines?
            lines = _best_lines_for_points(points, 27, max_time=10.0)
            count = _count_incidents(points, lines)
            
    return points, count


def _validate(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.allclose(points, np.round(points), atol=1e-6) or not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    pt_set = set()
    for p in points:
        t = (int(p[0]), int(p[1]))
        if t in pt_set:
            return False
        pt_set.add(t)
    line_set = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        nl = _normalize_line(A, B, C)
        if nl in line_set:
            return False
        line_set.add(nl)
    return True


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_lines = None
    best_count = -1
    
    # Try different grid configurations
    # 6x9, 9x6, 3x18, 18x3, 2x27, 27x2
    # 54 is not a square.
    # 7x8 = 56 (remove 2)
    # 5x11 = 55 (remove 1)
    # 4x14 = 56 (remove 2)
    
    # First, try exact grid 6x9
    p, l = _try_grid_configuration(6, 9, start_time, deadline)
    if p is not None:
        best_points = p
        best_lines = l
        best_count = _count_incidents(p, l)
        
    # Try 9x6
    p, l = _try_grid_configuration(9, 6, start_time, deadline)
    if p is not None:
        c = _count_incidents(p, l)
        if c > best_count:
            best_points = p
            best_lines = l
            best_count = c
            
    # Try 7x8 with 2 points removed
    # We need to try different removal patterns.
    # Common pattern: remove corners? Or random?
    # Let's try removing (0,0) and (6,7) (corners of 7x8)
    # And (0,7) and (6,0)
    # And (0,0) and (0,7)
    
    for removals in [((0,0), (6,7)), ((0,7), (6,0)), ((0,0), (0,7)), ((3,3), (3,4))]:
        if time.time() > deadline - 20:
            break
            
        points_x = np.arange(7)
        points_y = np.arange(8)
        px, py = np.meshgrid(points_x, points_y, indexing='ij')
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Remove points
        mask = np.ones(56, dtype=bool)
        for (r, c) in removals:
            idx = r * 8 + c
            mask[idx] = False
            
        points = points[mask]
        
        if len(points) != 54:
            continue
            
        lines = _best_lines_for_points(points, 27, max_time=10.0)
        count = _count_incidents(points, lines)
        
        # Local search
        if time.time() < deadline - 10:
            pts_ls, count_ls = _local_search_points(points, lines, max_time=min(30.0, deadline - time.time() - 5))
            if count_ls > count:
                points = pts_ls
                count = count_ls
                lines = _best_lines_for_points(points, 27, max_time=10.0)
                count = _count_incidents(points, lines)
                
        if count > best_count:
            best_points = points
            best_lines = lines
            best_count = count
            
    # Try 5x11 with 1 point removed
    for r in range(5):
        for c in range(11):
            if time.time() > deadline - 10:
                break
                
            points_x = np.arange(5)
            points_y = np.arange(11)
            px, py = np.meshgrid(points_x, points_y, indexing='ij')
            points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            
            mask = np.ones(55, dtype=bool)
            idx = r * 11 + c
            mask[idx] = False
            points = points[mask]
            
            if len(points) != 54:
                continue
                
            lines = _best_lines_for_points(points, 27, max_time=5.0)
            count = _count_incidents(points, lines)
            
            if count > best_count:
                best_points = points
                best_lines = lines
                best_count = count
                
            # Local search if promising
            if count > best_count - 5 and time.time() < deadline - 10:
                pts_ls, count_ls = _local_search_points(points, lines, max_time=min(20.0, deadline - time.time() - 5))
                if count_ls > best_count:
                    best_points = pts_ls
                    best_count = count_ls
                    best_lines = _best_lines_for_points(pts_ls, 27, max_time=10.0)
                    best_count = _count_incidents(best_points, best_lines)
                    
        if time.time() > deadline - 10:
            break

    # Final local search on the best configuration found
    if best_points is not None and time.time() < deadline - 5:
        pts_ls, count_ls = _local_search_points(best_points, best_lines, max_time=min(60.0, deadline - time.time() - 5))
        if count_ls > best_count:
            best_points = pts_ls
            best_count = count_ls
            best_lines = _best_lines_for_points(best_points, 27, max_time=10.0)
            best_count = _count_incidents(best_points, best_lines)
            
    # Validate
    if not _validate(best_points, best_lines):
        # Fallback to 6x9
        points_x = np.arange(6)
        points_y = np.arange(9)
        px, py = np.meshgrid(points_x, points_y, indexing='ij')
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = _best_lines_for_points(best_points, 27, max_time=5.0)
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
