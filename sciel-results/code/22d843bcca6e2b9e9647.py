# EVOLVE-BLOCK-START
import numpy as np
import time
import math


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
    # Make first nonzero entry positive
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


def _line_incidence_count(points, line):
    """Count how many points lie on the given line."""
    A, B, C = line
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))


def _generate_candidate_lines(points, deadline):
    """
    Generate candidate lines from point pairs and slope families.
    Returns a dict: line_tuple -> incidence_count
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
    
    # Additional slope families for grid-like structures
    # Horizontal lines
    for y in set(points[:, 1]):
        line = _normalize_line(0, 1, -int(y))
        if line not in line_counts:
            line_counts[line] = _line_incidence_count(points, line)
    
    # Vertical lines
    for x in set(points[:, 0]):
        line = _normalize_line(1, 0, -int(x))
        if line not in line_counts:
            line_counts[line] = _line_incidence_count(points, line)
    
    # Diagonal lines with various slopes
    slopes = [(1, 1), (-1, 1), (2, 1), (1, 2), (-2, 1), (1, -2), 
              (3, 1), (1, 3), (-3, 1), (1, -3), (2, -1), (-2, -1),
              (3, 2), (2, 3), (-3, 2), (3, -2)]
    
    for dx, dy in slopes:
        # For each point, the line with this slope through it
        seen_lines = set()
        for i in range(n):
            x, y = int(points[i, 0]), int(points[i, 1])
            # Line: dy*x - dx*y + C = 0, so C = dx*y - dy*x
            C = dx * y - dy * x
            line = _normalize_line(dy, -dx, C)
            if line not in seen_lines and line not in line_counts:
                line_counts[line] = _line_incidence_count(points, line)
                seen_lines.add(line)
    
    return line_counts


def _greedy_select_lines(points, line_counts, num_lines=27, deadline=None):
    """
    Greedily select lines to maximize total incidences.
    Uses a more sophisticated greedy that considers marginal gains.
    """
    if deadline is not None and time.time() > deadline:
        return []
    
    n = len(points)
    # Start with all lines sorted by their individual incidence count
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    selected = []
    selected_set = set()
    # Track which points are covered by how many selected lines
    point_coverage = np.zeros(n, dtype=np.int64)
    
    # Pre-compute for each line which points it covers
    line_point_sets = {}
    for line in sorted_lines:
        A, B, C = line[0]
        mask = (A * points[:, 0] + B * points[:, 1] + C) == 0
        line_point_sets[line[0]] = mask
    
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        if deadline is not None and time.time() > deadline:
            break
        if line in selected_set:
            continue
        
        # Calculate marginal gain: new incidences from this line
        mask = line_point_sets[line]
        # Points not yet covered by any selected line get full credit
        # Points already covered get less credit (diminishing returns)
        marginal = np.sum(mask * (1 + point_coverage * 0.5))
        
        if marginal > 0:
            selected.append(line)
            selected_set.add(line)
            point_coverage += mask
    
    # If we don't have enough, add more lines
    if len(selected) < num_lines:
        for line, count in sorted_lines:
            if len(selected) >= num_lines:
                break
            if line not in selected_set:
                selected.append(line)
                selected_set.add(line)
    
    return selected[:num_lines]


def _try_configuration(points, deadline):
    """Try a configuration and return (points, lines, inc)."""
    if deadline is not None and time.time() > deadline:
        return None
    
    line_counts = _generate_candidate_lines(points, deadline)
    if not line_counts:
        return None
    
    lines = _greedy_select_lines(points, line_counts, 27, deadline)
    if not lines or len(lines) < 27:
        return None
    
    lines_arr = np.array(lines[:27], dtype=np.int64)
    inc = _incidence_count(points, lines_arr)
    return points.copy(), lines_arr, inc


def _local_search(points, lines, deadline, max_iters=200):
    """
    Local search: try moving individual points and re-optimizing lines.
    """
    best_points = points.copy()
    best_lines = lines.copy()
    best_inc = _incidence_count(best_points, best_lines)
    n = len(best_points)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try moving each point
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_points = best_points.copy()
                    new_points[i, 0] += dx
                    new_points[i, 1] += dy
                    
                    # Check for duplicates
                    if len(np.unique(new_points, axis=0)) != n:
                        continue
                    
                    # Check for non-negative or reasonable coordinates
                    if np.any(new_points < -100) or np.any(new_points > 100):
                        continue
                    
                    # Re-optimize lines for new points
                    line_counts = _generate_candidate_lines(new_points, deadline)
                    if not line_counts:
                        continue
                    
                    new_lines = _greedy_select_lines(new_points, line_counts, 27, deadline)
                    if not new_lines or len(new_lines) < 27:
                        continue
                    
                    new_lines_arr = np.array(new_lines[:27], dtype=np.int64)
                    new_inc = _incidence_count(new_points, new_lines_arr)
                    
                    if new_inc > best_inc:
                        best_points = new_points
                        best_lines = new_lines_arr
                        best_inc = new_inc
                        improved = True
        
        if not improved:
            break
    
    return best_points, best_lines, best_inc


def _generate_grid_points(kx, ky, spacing=1):
    """Generate a kx by ky grid of points."""
    xs = np.arange(1, kx + 1) * spacing
    ys = np.arange(1, ky + 1) * spacing
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _generate_staggered_grid(kx, ky, offset=1):
    """Generate a staggered grid."""
    points = []
    for j in range(ky):
        for i in range(kx):
            x = i + (j % 2) * offset
            y = j
            points.append([x, y])
    return np.array(points, dtype=np.int64)


def _generate_hexagonal_grid(n_points):
    """Generate a hexagonal-like arrangement of points."""
    points = []
    # Hexagonal packing
    row = 0
    while len(points) < n_points:
        for col in range(n_points):
            x = col + (row % 2) * 0.5
            y = row * 0.866
            if len(points) < n_points:
                points.append([int(round(x * 2)), int(round(y * 2))])
        row += 1
        if len(points) >= n_points:
            break
    
    # Ensure unique points
    unique_points = []
    seen = set()
    for p in points[:n_points]:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique_points.append(list(key))
    
    # Pad if needed
    while len(unique_points) < n_points:
        unique_points.append([len(unique_points), 0])
    
    return np.array(unique_points[:n_points], dtype=np.int64)


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 second safety margin
    
    best_points = None
    best_lines = None
    best_inc = 0
    
    # Try different grid configurations
    grid_configs = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)
    ]
    
    for kx, ky in grid_configs:
        if time.time() > deadline:
            break
        points = _generate_grid_points(kx, ky)
        result = _try_configuration(points, deadline)
        if result is not None:
            pts, lines, inc = result
            if inc > best_inc:
                best_points = pts
                best_lines = lines
                best_inc = inc
    
    # Try staggered grids
    for kx, ky in [(6, 9), (9, 6), (3, 18), (18, 3)]:
        if time.time() > deadline:
            break
        for offset in [1, 2]:
            points = _generate_staggered_grid(kx, ky, offset)
            if len(points) != 54:
                continue
            result = _try_configuration(points, deadline)
            if result is not None:
                pts, lines, inc = result
                if inc > best_inc:
                    best_points = pts
                    best_lines = lines
                    best_inc = inc
    
    # Try hexagonal
    if time.time() < deadline - 30:
        points = _generate_hexagonal_grid(54)
        result = _try_configuration(points, deadline)
        if result is not None:
            pts, lines, inc = result
            if inc > best_inc:
                best_points = pts
                best_lines = lines
                best_inc = inc
    
    # If we have a result, try local search
    if best_points is not None and time.time() < deadline - 60:
        best_points, best_lines, best_inc = _local_search(
            best_points, best_lines, deadline, max_iters=100
        )
    
    # Fallback: simple grid
    if best_points is None:
        kx, ky = 6, 9
        best_points = _generate_grid_points(kx, ky)
        line_counts = _generate_candidate_lines(best_points, deadline)
        best_lines_list = _greedy_select_lines(best_points, line_counts, 27, deadline)
        if not best_lines_list or len(best_lines_list) < 27:
            # Create default lines
            best_lines_list = []
            for x in range(1, kx + 1):
                best_lines_list.append((1, 0, -x))
            for y in range(1, ky + 1):
                best_lines_list.append((0, 1, -y))
            while len(best_lines_list) < 27:
                best_lines_list.append((1, 1, 0))
        best_lines = np.array(best_lines_list[:27], dtype=np.int64)
        best_inc = _incidence_count(best_points, best_lines)
    
    # Verify the result
    final_incidents = _incidence_count(best_points, best_lines)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
