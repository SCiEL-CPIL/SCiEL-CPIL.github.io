# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    px = points[:, 0:1]  # (n, 1)
    py = points[:, 1:2]  # (n, 1)
    A = lines[0:1, :]    # (1, m)
    B = lines[1:2, :]    # (1, m)
    C = lines[2:3, :]    # (1, m)
    total = px * A + py * B + C  # (n, m)
    return int(np.sum(total == 0))


def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_two_points(p1, p2):
    """Get line through two points as (A, B, C) normalized."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    if x1 == x2:
        return _normalize_line(1, 0, -x1)
    elif y1 == y2:
        return _normalize_line(0, 1, -y1)
    else:
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return _normalize_line(A, B, C)


def _line_incidence_mask(points, line):
    """Return boolean mask of points on the given line."""
    A, B, C = line
    vals = A * points[:, 0] + B * points[:, 1] + C
    return (vals == 0)


def _generate_candidate_lines(points, deadline=None):
    """
    Generate candidate lines from point pairs and slope families.
    Returns list of (line_tuple, incidence_count)
    """
    n = len(points)
    line_counts = {}
    
    # Lines through all pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_two_points(points[i], points[j])
            if line not in line_counts:
                line_counts[line] = 0
            line_counts[line] += 1
    
    # Horizontal lines
    for y in set(points[:, 1]):
        line = _normalize_line(0, 1, -int(y))
        if line not in line_counts:
            line_counts[line] = int(np.sum(_line_incidence_mask(points, line)))
    
    # Vertical lines
    for x in set(points[:, 0]):
        line = _normalize_line(1, 0, -int(x))
        if line not in line_counts:
            line_counts[line] = int(np.sum(_line_incidence_mask(points, line)))
    
    # Diagonal lines with various slopes
    slopes = [(1, 1), (-1, 1), (2, 1), (1, 2), (-2, 1), (1, -2), 
              (3, 1), (1, 3), (-3, 1), (1, -3), (2, -1), (-2, -1),
              (3, 2), (2, 3), (-3, 2), (3, -2), (4, 1), (1, 4),
              (-4, 1), (1, -4), (4, -1), (-4, -1)]
    
    for dx, dy in slopes:
        seen_lines = set()
        for i in range(n):
            x, y = int(points[i, 0]), int(points[i, 1])
            C = dx * y - dy * x
            line = _normalize_line(dy, -dx, C)
            if line not in seen_lines:
                seen_lines.add(line)
                if line not in line_counts:
                    line_counts[line] = int(np.sum(_line_incidence_mask(points, line)))
    
    return line_counts


def _greedy_submodular_select(points, line_counts, num_lines=27, deadline=None):
    """
    Greedy submodular maximization: select lines maximizing total incidences
    considering point overlaps (each point counted once per line it's on).
    """
    if deadline is not None and time.time() > deadline:
        return []
    
    n = len(points)
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Pre-compute masks
    line_masks = {}
    for line in sorted_lines:
        line_masks[line[0]] = _line_incidence_mask(points, line[0])
    
    selected = []
    selected_set = set()
    point_coverage = np.zeros(n, dtype=np.int64)
    
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        if deadline is not None and time.time() > deadline:
            break
        if line in selected_set:
            continue
        
        mask = line_masks[line]
        # Marginal gain: points on this line that aren't already covered get full credit
        # Points already covered get half credit (diminishing returns)
        marginal = np.sum(mask * (1.0 + 0.3 * point_coverage))
        
        if marginal > 0:
            selected.append(line)
            selected_set.add(line)
            point_coverage += mask.astype(np.int64)
    
    # Fill remaining with highest individual count
    if len(selected) < num_lines:
        for line, count in sorted_lines:
            if len(selected) >= num_lines:
                break
            if line not in selected_set:
                selected.append(line)
                selected_set.add(line)
    
    return selected[:num_lines]


def _fast_point_optimization(points, lines, deadline, max_rounds=50):
    """
    Fast coordinate descent: try moving each point to a nearby position
    that maximizes total incidence with the fixed lines.
    """
    best_points = points.copy()
    best_inc = _incidence_count(best_points, lines)
    n = len(best_points)
    m = len(lines)
    
    # Pre-compute line coefficients
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            bx, by = int(best_points[i, 0]), int(best_points[i, 1])
            best_local_inc = best_inc
            best_bx, best_by = bx, by
            
            # Try a range of positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = bx + dx, by + dy
                    
                    # Quick check: compute incidences for this point with all lines
                    vals = A * nx + B * ny + C
                    new_point_inc = int(np.sum(vals == 0))
                    
                    # Total = best_inc - old_point_inc + new_point_inc
                    old_vals = A * bx + B * by + C
                    old_point_inc = int(np.sum(old_vals == 0))
                    
                    new_total = best_inc - old_point_inc + new_point_inc
                    
                    if new_total > best_local_inc:
                        # Check for duplicates
                        dup = False
                        for j in range(n):
                            if j != i and best_points[j, 0] == nx and best_points[j, 1] == ny:
                                dup = True
                                break
                        if not dup:
                            best_local_inc = new_total
                            best_bx, best_by = nx, ny
            
            if best_bx != bx or best_by != by:
                best_points[i, 0] = best_bx
                best_points[i, 1] = best_by
                best_inc = best_local_inc
                improved = True
        
        if not improved:
            break
    
    return best_points, best_inc


def _generate_parabola_points(n_pts):
    """Generate points on a parabola y = x^2 / k for various k."""
    points = []
    for k in [1, 2, 3]:
        for x in range(0, n_pts // 3 + 1):
            if len(points) >= n_pts:
                break
            y = x * x // k
            points.append([x, y])
    # Fill remaining
    seen = set(tuple(p) for p in points)
    for x in range(-20, 20):
        for y in range(-20, 20):
            if len(points) >= n_pts:
                break
            if (x, y) not in seen:
                points.append([x, y])
                seen.add((x, y))
    return np.array(points[:n_pts], dtype=np.int64)


def _generate_modular_points(n_pts):
    """Generate points based on modular arithmetic patterns."""
    points = []
    for a in range(1, 10):
        for b in range(0, 10):
            if len(points) >= n_pts:
                break
            x = a * 3
            y = (a * a + b) % 15
            points.append([x, y])
    # Ensure unique
    unique = []
    seen = set()
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    while len(unique) < n_pts:
        unique.append([len(unique) * 2, len(unique) * 3])
    return np.array(unique[:n_pts], dtype=np.int64)


def _generate_diamond_points(n_pts):
    """Generate points in a diamond/rhombus pattern."""
    points = []
    for r in range(0, 10):
        for i in range(-r, r + 1):
            if len(points) >= n_pts:
                break
            x = i
            y = r - abs(i)
            points.append([x + 5, y + 5])
    unique = []
    seen = set()
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    while len(unique) < n_pts:
        unique.append([len(unique) + 10, len(unique) + 20])
    return np.array(unique[:n_pts], dtype=np.int64)


def _evaluate_and_refine(points, deadline):
    """Evaluate a point configuration and refine it."""
    if deadline is not None and time.time() > deadline:
        return None
    
    line_counts = _generate_candidate_lines(points, deadline)
    if not line_counts:
        return None
    
    lines = _greedy_submodular_select(points, line_counts, 27, deadline)
    if not lines or len(lines) < 27:
        return None
    
    lines_arr = np.array(lines[:27], dtype=np.int64)
    inc = _incidence_count(points, lines_arr)
    
    # Fast point optimization
    if deadline is not None and time.time() < deadline - 5:
        points, inc = _fast_point_optimization(points, lines_arr, deadline, max_rounds=30)
        # Re-optimize lines after point movement
        line_counts2 = _generate_candidate_lines(points, deadline)
        lines2 = _greedy_submodular_select(points, line_counts2, 27, deadline)
        if lines2 and len(lines2) >= 27:
            lines_arr = np.array(lines2[:27], dtype=np.int64)
            inc2 = _incidence_count(points, lines_arr)
            if inc2 > inc:
                inc = inc2
    
    return points.copy(), lines_arr, inc


def _generate_grid_points(kx, ky, spacing=1):
    """Generate a kx by ky grid of points."""
    xs = np.arange(1, kx + 1) * spacing
    ys = np.arange(1, ky + 1) * spacing
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 580  # 50 second safety margin
    
    best_points = None
    best_lines = None
    best_inc = 0
    
    # Try different grid configurations
    grid_configs = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1),
        (4, 14), (14, 4), (5, 11), (11, 5)
    ]
    
    for kx, ky in grid_configs:
        if time.time() > deadline:
            break
        if kx * ky != 54:
            continue
        points = _generate_grid_points(kx, ky)
        result = _evaluate_and_refine(points, deadline)
        if result is not None:
            pts, lines, inc = result
            if inc > best_inc:
                best_points = pts
                best_lines = lines
                best_inc = inc
    
    # Try structured configurations
    structured_generators = [
        _generate_parabola_points,
        _generate_modular_points,
        _generate_diamond_points,
    ]
    
    for gen in structured_generators:
        if time.time() > deadline:
            break
        points = gen(54)
        if len(points) != 54:
            continue
        result = _evaluate_and_refine(points, deadline)
        if result is not None:
            pts, lines, inc = result
            if inc > best_inc:
                best_points = pts
                best_lines = lines
                best_inc = inc
    
    # Random restarts with diverse initializations
    random.seed(42)
    for restart in range(15):
        if time.time() > deadline:
            break
        
        # Random points in a bounded region
        points = np.random.randint(-15, 16, size=(54, 2), dtype=np.int64)
        # Ensure uniqueness
        unique_pts = []
        seen = set()
        for p in points:
            key = (int(p[0]), int(p[1]))
            if key not in seen:
                seen.add(key)
                unique_pts.append(list(key))
        while len(unique_pts) < 54:
            new_p = [random.randint(-15, 15), random.randint(-15, 15)]
            if (new_p[0], new_p[1]) not in seen:
                seen.add((new_p[0], new_p[1]))
                unique_pts.append(new_p)
        points = np.array(unique_pts[:54], dtype=np.int64)
        
        result = _evaluate_and_refine(points, deadline)
        if result is not None:
            pts, lines, inc = result
            if inc > best_inc:
                best_points = pts
                best_lines = lines
                best_inc = inc
    
    # Final polish on best result
    if best_points is not None and time.time() < deadline - 10:
        best_points, best_inc = _fast_point_optimization(
            best_points, best_lines, deadline, max_rounds=20
        )
        line_counts = _generate_candidate_lines(best_points, deadline)
        lines_final = _greedy_submodular_select(best_points, line_counts, 27, deadline)
        if lines_final and len(lines_final) >= 27:
            best_lines = np.array(lines_final[:27], dtype=np.int64)
            best_inc = _incidence_count(best_points, best_lines)
    
    # Fallback
    if best_points is None:
        best_points = _generate_grid_points(6, 9)
        line_counts = _generate_candidate_lines(best_points, deadline)
        best_lines_list = _greedy_submodular_select(best_points, line_counts, 27, deadline)
        if not best_lines_list or len(best_lines_list) < 27:
            best_lines_list = []
            for x in range(1, 7):
                best_lines_list.append((1, 0, -x))
            for y in range(1, 10):
                best_lines_list.append((0, 1, -y))
            while len(best_lines_list) < 27:
                best_lines_list.append((1, 1, 0))
        best_lines = np.array(best_lines_list[:27], dtype=np.int64)
        best_inc = _incidence_count(best_points, best_lines)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
