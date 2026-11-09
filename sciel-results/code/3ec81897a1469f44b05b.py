# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations

def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    a, b, c = abs(a), abs(b), abs(c)
    if a == 0 and b == 0 and c == 0:
        return 1
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g if g != 0 else 1

def _normalize_line(A, B, C):
    """Normalize a line so it's canonical for comparison."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    """Count incidences: number of (point, line) pairs where A*x + B*y + C == 0."""
    # points: (n, 2) array
    # lines: (m, 3) array
    # For each line, count how many points lie on it
    # Vectorized: for line (A, B, C), check A*x + B*y + C == 0 for all points
    total = 0
    for A, B, C in lines:
        # A*px + B*py + C == 0
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.count_nonzero(vals == 0)
    return int(total)

def _validate(points, lines):
    """Validate the configuration. Returns True if valid."""
    # Check shapes
    if points.shape != (54, 2):
        return False
    if lines.shape != (27, 3):
        return False
    
    # Check for NaN/inf
    if not np.all(np.isfinite(points)):
        return False
    if not np.all(np.isfinite(lines)):
        return False
    
    # Check entries are close to integers
    if not np.all(np.abs(points - np.round(points)) < 1e-6):
        return False
    if not np.all(np.abs(lines - np.round(lines)) < 1e-6):
        return False
    
    # Round to int64
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    # Check for repeated points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != 54:
        return False
    
    # Check lines: A=B=0 is invalid
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
    
    # Check for duplicate lines (after normalization)
    norm_lines = set()
    for A, B, C in lines:
        nl = _normalize_line(int(A), int(B), int(C))
        if nl in norm_lines:
            return False
        norm_lines.add(nl)
    
    return True

def _generate_candidate_lines_from_points(points, max_slopes=20):
    """
    Generate candidate lines that pass through at least 2 points.
    For each pair of points, compute the line, then count how many points lie on it.
    Return lines sorted by incidence count (descending).
    """
    n = len(points)
    line_incidents = {}  # normalized line -> count
    
    # For each pair of points, compute the line
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through (x1,y1) and (x2,y2):
            # (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            
            if A == 0 and B == 0:
                continue
            
            nl = _normalize_line(A, B, C)
            
            if nl not in line_incidents:
                # Count how many points lie on this line
                count = 0
                for k in range(n):
                    xk, yk = points[k]
                    if A * xk + B * yk + C == 0:
                        count += 1
                line_incidents[nl] = count
    
    # Sort by incidence count descending
    sorted_lines = sorted(line_incidents.items(), key=lambda x: -x[1])
    
    return sorted_lines

def _build_grid_and_lines(rows, cols):
    """Build a grid of points and candidate lines."""
    xs = np.arange(1, rows + 1)
    ys = np.arange(1, cols + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate all candidate lines from point pairs
    sorted_lines = _generate_candidate_lines_from_points(points)
    
    return points, sorted_lines

def _local_search(points, initial_line_indices, sorted_lines, num_lines=27, max_iterations=5000):
    """
    Local search: start with initial line indices, try to swap in better lines.
    sorted_lines: list of (normalized_line, incidence_count)
    """
    n_lines = len(sorted_lines)
    current = set(initial_line_indices)
    current_incidents = _count_incidents(points, np.array([sorted_lines[i][0] for i in current], dtype=np.int64))
    best_incidents = current_incidents
    best_lines = set(current)
    
    # Precompute incidence contribution of each line
    line_counts = [sorted_lines[i][1] for i in range(n_lines)]
    
    for it in range(max_iterations):
        improved = False
        # Try swapping each current line with a non-current line
        for i in list(current):
            for j in range(n_lines):
                if j in current:
                    continue
                new_set = (current - {i}) | {j}
                # Estimate: remove contribution of line i, add contribution of line j
                # But this is an overestimate since lines can share points
                # For speed, only do full count if the estimate looks promising
                est = current_incidents - line_counts[i] + line_counts[j]
                if est > best_incidents:
                    new_incidents = _count_incidents(points, np.array([sorted_lines[k][0] for k in new_set], dtype=np.int64))
                    if new_incidents > best_incidents:
                        best_incidents = new_incidents
                        best_lines = new_set
                        improved = True
                        current = new_set
                        current_incidents = new_incidents
        if not improved:
            break
    
    return best_lines, best_incidents

def _simulated_annealing(points, sorted_lines, num_lines=27, max_time=580.0):
    """
    Simulated annealing over line subsets.
    sorted_lines: list of (normalized_line, incidence_count)
    """
    n_lines = len(sorted_lines)
    start_time = time.time()
    
    # Initial solution: top num_lines lines by individual incidence
    initial = list(range(num_lines))
    current = set(initial)
    current_incidents = _count_incidents(points, np.array([sorted_lines[i][0] for i in current], dtype=np.int64))
    best = set(current)
    best_incidents = current_incidents
    
    # Precompute line incidence counts
    line_counts = [sorted_lines[i][1] for i in range(n_lines)]
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling = 0.999
    
    T = T_start
    step = 0
    while T > T_end:
        elapsed = time.time() - start_time
        if elapsed > max_time - 10:  # safety margin
            break
        
        step += 1
        
        # Choose a random move: remove one line, add another
        if len(current) == 0:
            break
        
        i = list(current)[np.random.randint(len(current))]
        j = np.random.randint(n_lines)
        if j in current:
            continue
        
        new_set = (current - {i}) | {j}
        
        # Compute new incidence count
        new_incidents = _count_incidents(points, np.array([sorted_lines[k][0] for k in new_set], dtype=np.int64))
        delta = new_incidents - current_incidents
        
        if delta >= 0 or np.random.random() < math.exp(delta / T):
            current = new_set
            current_incidents = new_incidents
            if current_incidents > best_incidents:
                best = set(current)
                best_incidents = current_incidents
        
        T *= cooling
    
    return best, best_incidents

def _try_configuration(points, sorted_lines, max_time=580.0):
    """Try multiple strategies to find the best line subset."""
    start_time = time.time()
    num_lines = 27
    
    # Strategy 1: Greedy - pick top 27 lines by individual incidence
    greedy = list(range(min(num_lines, len(sorted_lines))))
    greedy_incidents = _count_incidents(points, np.array([sorted_lines[i][0] for i in greedy], dtype=np.int64))
    best_lines = set(greedy)
    best_incidents = greedy_incidents
    
    # Strategy 2: Local search from greedy
    if time.time() - start_time < max_time - 60:
        ls_lines, ls_incidents = _local_search(points, greedy, sorted_lines, num_lines, max_iterations=2000)
        if ls_incidents > best_incidents:
            best_lines = ls_lines
            best_incidents = ls_incidents
    
    # Strategy 3: Simulated annealing
    if time.time() - start_time < max_time - 30:
        sa_lines, sa_incidents = _simulated_annealing(points, sorted_lines, num_lines, max_time=max_time - 15)
        if sa_incidents > best_incidents:
            best_lines = sa_lines
            best_incidents = sa_incidents
    
    return best_lines, best_incidents

def _try_grid(rows, cols, max_time=580.0):
    """Try a specific grid configuration."""
    if rows * cols != 54:
        return None
    
    points, sorted_lines = _build_grid_and_lines(rows, cols)
    
    if len(sorted_lines) < 27:
        return None
    
    best_lines, best_incidents = _try_configuration(points, sorted_lines, max_time)
    
    if best_lines is None or len(best_lines) != 27:
        return None
    
    lines_arr = np.array([sorted_lines[i][0] for i in sorted(best_lines)], dtype=np.int64)
    
    return points, lines_arr, best_incidents

def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines maximizing incidences.
    Uses simulated annealing over grid-based configurations.
    """
    start_time = time.time()
    max_time = 590.0  # safety margin before 600s limit
    
    # Try different grid configurations
    # 54 = 1*54, 2*27, 3*18, 6*9, 9*6, 18*3, 27*2, 54*1
    # Focus on more balanced grids for better line structure
    grids_to_try = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]
    
    best_result = None
    best_score = -1
    
    for rows, cols in grids_to_try:
        elapsed = time.time() - start_time
        if elapsed > max_time - 60:
            break
        
        time_budget = max_time - 60 - elapsed
        if time_budget < 30:
            break
        
        result = _try_grid(rows, cols, max_time=time_budget)
        
        if result is not None:
            points, lines, score = result
            if score > best_score:
                best_score = score
                best_result = (points, lines)
    
    if best_result is None:
        # Fallback: basic 6x9 grid with vertical, horizontal, diagonal lines
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        for c in [0, 1, 2, 3, -1, 4]:
            lines.append((1, -1, c))
        for c in [7, 8, 9, 10, 6, 11]:
            lines.append((1, 1, -c))
        
        lines = np.array(lines, dtype=np.int64)
        best_result = (points, lines)
    
    points, lines = best_result
    
    # Final validation
    if not _validate(points, lines):
        # Fallback to known valid configuration
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        for c in [0, 1, 2, 3, -1, 4]:
            lines.append((1, -1, c))
        for c in [7, 8, 9, 10, 6, 11]:
            lines.append((1, 1, -c))
        
        lines = np.array(lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
