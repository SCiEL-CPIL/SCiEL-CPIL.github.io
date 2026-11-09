import numpy as np
import math
import time
import random

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) to a canonical form."""
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _get_all_lines_through_pairs(points):
    """Generate all unique lines through pairs of points, with incidence counts."""
    n_pts = len(points)
    line_map = {}
    
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            if x1 == x2:
                line = (1, 0, -x1)
            elif y1 == y2:
                line = (0, 1, -y1)
            else:
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                line = (A, B, C)
            
            nline = _normalize_line(*line)
            if nline not in line_map:
                line_map[nline] = line
    
    # Count incidences for each unique line
    line_counts = {}
    for nline, (A, B, C) in line_map.items():
        count = 0
        for i in range(n_pts):
            x, y = points[i]
            if A * x + B * y + C == 0:
                count += 1
        line_counts[nline] = count
    
    return line_map, line_counts

def _select_best_lines(line_map, line_counts, num_lines=27):
    """Select best lines by incidence count."""
    sorted_lines = sorted(line_counts.items(), key=lambda x: (-x[1], x[0]))
    selected = [line_map[nline] for nline, _ in sorted_lines[:num_lines]]
    return selected

def _calculate_incidence(points, lines):
    """Calculate total incidences for given points and lines."""
    I = 0
    for (A, B, C) in lines:
        for (x, y) in points:
            if A * x + B * y + C == 0:
                I += 1
    return I

def _initial_grid_config():
    """Create initial 6x9 grid configuration."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64).copy()
    return points

def _initial_diagonal_config():
    """Create initial configuration with diagonal structure."""
    # Use a more symmetric starting point
    points = []
    for i in range(6):
        for j in range(9):
            points.append([i, j])
    return np.array(points, dtype=np.int64)

def _simulated_annealing(points, deadline):
    """
    Simulated annealing to optimize point positions for maximum incidences.
    """
    random.seed(42)
    
    current_points = points.copy()
    current_lines = _select_best_lines(*_get_all_lines_through_pairs(current_points), 27)
    current_score = _calculate_incidence(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines
    best_score = current_score
    
    # Annealing parameters
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    # Precompute occupied set for fast collision checking
    occupied_set = set(map(tuple, current_points))
    
    iterations = 0
    while time.time() < deadline and T > T_min:
        iterations += 1
        
        # Pick a random point to move
        idx = random.randint(0, 53)
        orig_x, orig_y = current_points[idx]
        
        # Generate a random perturbation with varying magnitude
        # Larger moves early, smaller moves late
        max_move = max(1, int(3 * T))
        dx = random.randint(-max_move, max_move)
        dy = random.randint(-max_move, max_move)
        
        if dx == 0 and dy == 0:
            continue
            
        new_x = int(orig_x) + dx
        new_y = int(orig_y) + dy
        
        # Check collision
        new_pos = (new_x, new_y)
        if new_pos in occupied_set:
            continue
            
        # Apply move
        current_points[idx] = [new_x, new_y]
        occupied_set.remove((orig_x, orig_y))
        occupied_set.add(new_pos)
        
        # Recalculate score
        new_lines = _select_best_lines(*_get_all_lines_through_pairs(current_points), 27)
        new_score = _calculate_incidence(current_points, new_lines)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta >= 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            current_lines = new_lines
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                best_lines = current_lines
        else:
            # Revert
            current_points[idx] = [orig_x, orig_y]
            occupied_set.remove(new_pos)
            occupied_set.add((orig_x, orig_y))
        
        # Cool down
        T *= cooling_rate
    
    return best_points, best_lines, best_score

def _try_multiple_starts(deadline):
    """Try multiple initial configurations with simulated annealing."""
    random.seed(42)
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Start 1: Standard grid
    points1 = _initial_grid_config()
    bp1, bl1, bs1 = _simulated_annealing(points1, deadline)
    if bs1 > best_score:
        best_points, best_lines, best_score = bp1, bl1, bs1
    
    # Start 2: Slightly perturbed grid
    points2 = _initial_grid_config().copy()
    for i in range(54):
        points2[i][0] += random.randint(-1, 1)
        points2[i][1] += random.randint(-1, 1)
    # Remove duplicates
    seen = set()
    unique_points = []
    for p in points2:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    # If we lost points, add new ones
    while len(unique_points) < 54:
        x = random.randint(-2, 7)
        y = random.randint(-2, 10)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_points.append([x, y])
    points2 = np.array(unique_points[:54], dtype=np.int64)
    
    if time.time() < deadline:
        bp2, bl2, bs2 = _simulated_annealing(points2, deadline)
        if bs2 > best_score:
            best_points, best_lines, best_score = bp2, bl2, bs2
    
    # Start 3: Different grid orientation
    xs = np.arange(9)
    ys = np.arange(6)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points3 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64).copy()
    
    if time.time() < deadline:
        bp3, bl3, bs3 = _simulated_annealing(points3, deadline)
        if bs3 > best_score:
            best_points, best_lines, best_score = bp3, bl3, bs3
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Returns 54 points and 27 lines maximizing incidences.
    Uses simulated annealing with multiple starting configurations.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_points, best_lines, best_score = _try_multiple_starts(deadline)
    
    if best_points is None:
        # Fallback to simple grid
        points = _initial_grid_config()
        lines = _select_best_lines(*_get_all_lines_through_pairs(points), 27)
        return points, np.array(lines, dtype=np.int64)
    
    return best_points, np.array(best_lines, dtype=np.int64)
