# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _gcd3(a, b, c):
    return gcd(gcd(abs(a), abs(b)), abs(c))

def _normalize_line(A, B, C):
    g = _gcd3(A, B, C)
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    count = 0
    for i in range(lines.shape[0]):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            continue
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += int(np.sum(vals == 0))
    return count

def _validate(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lines.astype(np.float64))):
        return False
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    unique_points = np.unique(points, axis=0)
    if unique_points.shape[0] != 54:
        return False
    seen_lines = set()
    for i in range(lines.shape[0]):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _generate_all_candidate_lines(points):
    """Generate all lines through at least 2 points."""
    n = points.shape[0]
    lines_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            lines_set.add(norm)
    return list(lines_set)

def _greedy_line_selection(points, n_lines=27):
    """Greedy set-cover style: pick lines that maximize marginal incidence gain."""
    n = points.shape[0]
    candidates = _generate_all_candidate_lines(points)
    
    # Compute incidence for each candidate
    line_incidences = {}
    for line in candidates:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        line_incidences[line] = int(np.sum(vals == 0))
    
    selected = []
    selected_norms = set()
    total_incidents = 0
    point_line_count = np.zeros(n, dtype=np.int64)  # how many selected lines each point is on
    
    for _ in range(n_lines):
        best_line = None
        best_marginal = -1
        for line in candidates:
            if line in selected_norms:
                continue
            A, B, C = line
            vals = A * points[:, 0] + B * points[:, 1] + C
            on_line = vals == 0
            # Marginal gain: points that are on this line but not on any other selected line
            # Actually, total incidence = sum over all lines of (# points on that line)
            # So marginal gain of adding this line = (# points on this line)
            # But we want to maximize total, so just pick the one with most points
            gain = line_incidences[line]
            if gain > best_marginal:
                best_marginal = gain
                best_line = line
        
        if best_line is None:
            break
        selected.append(best_line)
        selected_norms.add(best_line)
        total_incidents += best_marginal
    
    # Pad with dummy lines if needed (shouldn't happen with 54 points)
    while len(selected) < n_lines:
        # Find a line not already selected
        for line in candidates:
            if line not in selected_norms:
                selected.append(line)
                selected_norms.add(line)
                break
        else:
            selected.append((1, 2, 999))
    
    return np.array(selected[:n_lines], dtype=np.int64), total_incidents

def _optimize_points_given_lines(lines, max_iter=50, time_limit=None):
    """Try to move individual points to increase total incidences."""
    start = time.time()
    n = 54
    # We'll try small perturbations: move each point by small offsets
    best_points = None
    best_score = -1
    points = None
    
    # Start with a reasonable initial point set based on line structure
    # For now, this is a placeholder - we need initial points
    return points, best_score

def _joint_optimize(points, lines, max_iter=200, time_limit=None):
    """Alternate between optimizing lines (given points) and points (given lines)."""
    start = time.time()
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidents(best_points, best_lines)
    
    for it in range(max_iter):
        if time_limit and time.time() > time_limit:
            break
        
        # Step 1: Optimize lines given points
        new_lines, _ = _greedy_line_selection(best_points, 27)
        new_score = _count_incidents(best_points, new_lines)
        if new_score > best_score:
            best_lines = new_lines
            best_score = new_score
        
        # Step 2: Try to improve points given lines
        # For each point, try small moves
        improved = False
        for i in range(54):
            if time_limit and time.time() > time_limit:
                break
            x, y = int(best_points[i, 0]), int(best_points[i, 1])
            best_move = (x, y)
            best_move_score = best_score
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    # Check if this creates a duplicate point
                    new_points = best_points.copy()
                    new_points[i] = [nx, ny]
                    if len(np.unique(new_points, axis=0)) != 54:
                        continue
                    score = _count_incidents(new_points, best_lines)
                    if score > best_move_score:
                        best_move_score = score
                        best_move = (nx, ny)
            
            if best_move != (x, y):
                best_points[i] = best_move
                best_score = best_move_score
                improved = True
        
        if not improved:
            # No point improvement, try line improvement again
            new_lines, _ = _greedy_line_selection(best_points, 27)
            new_score = _count_incidents(best_points, new_lines)
            if new_score > best_score:
                best_lines = new_lines
                best_score = new_score
            else:
                break
    
    return best_points, best_lines, best_score

def _build_grid_and_lines(gx, gy, n_diag=12):
    """Build a gx x gy grid with all vertical/horizontal lines and best diagonal lines."""
    xs = np.arange(1, gx + 1)
    ys = np.arange(1, gy + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # Vertical lines
    for i in range(1, gx + 1):
        lines.append((1, 0, -i))
    
    # Horizontal lines
    for j in range(1, gy + 1):
        lines.append((0, 1, -j))
    
    # Compute all possible diagonal lines and their incidence counts
    all_diag = []
    
    # Slope 1: y = x + c -> (1, -1, c)
    for c in range(-(gy - 1), gx):
        count = 0
        for x in range(1, gx + 1):
            y = x + c
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            all_diag.append((count, (1, -1, c)))
    
    # Slope -1: y = -x + c -> (1, 1, -c)
    for c in range(2, gx + gy + 1):
        count = 0
        for x in range(1, gx + 1):
            y = c - x
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            all_diag.append((count, (1, 1, -c)))
    
    # Sort by count descending
    all_diag.sort(key=lambda x: x[0], reverse=True)
    
    # Select top n_diag
    for count, line in all_diag[:n_diag]:
        lines.append(line)
    
    # Pad or trim to exactly 27 lines
    while len(lines) < 27:
        lines.append((1, 2, 1))
    
    lines = lines[:27]
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines

def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines maximizing incidences.
    Uses grid-based construction with joint optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_incidents = 0
    best_points = None
    best_lines = None
    
    # Grid configurations: (gx, gy) where gx*gy = 54
    configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
    ]
    
    for gx, gy in configs:
        if time.time() > deadline:
            break
        
        n_lines_axis = gx + gy
        n_diag = 27 - n_lines_axis
        if n_diag < 0:
            continue
        
        points, lines = _build_grid_and_lines(gx, gy, n_diag)
        
        if not _validate(points, lines):
            continue
        
        incidents = _count_incidents(points, lines)
        
        if incidents > best_incidents:
            best_incidents = incidents
            best_points = points.copy()
            best_lines = lines.copy()
        
        # Joint optimization
        remaining_time = deadline - time.time()
        if remaining_time > 30:
            opt_points, opt_lines, opt_score = _joint_optimize(
                points, lines,
                max_iter=100,
                time_limit=start_time + remaining_time - 10
            )
            if opt_score > best_incidents and _validate(opt_points, opt_lines):
                best_incidents = opt_score
                best_points = opt_points.copy()
                best_lines = opt_lines.copy()
    
    # If no valid configuration found, use fallback
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        c_vals_slope_1 = [0, 1, 2, 3, -1, 4]
        for c in c_vals_slope_1:
            lines.append((1, -1, c))
        c_vals_slope_neg_1 = [7, 8, 9, 10, 6, 11]
        for c in c_vals_slope_neg_1:
            lines.append((1, 1, -c))
        best_lines = np.array(lines, dtype=np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
