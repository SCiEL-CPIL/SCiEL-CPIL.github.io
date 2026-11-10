import numpy as np
import math
import time
import random
from collections import defaultdict

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

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _get_incidence_counts(points, lines):
    """
    Returns a list of incidence counts for each line, and a list of point indices on each line.
    points: (54, 2) array
    lines: list of (A, B, C) tuples
    """
    n_lines = len(lines)
    counts = [0] * n_lines
    point_sets = [set() for _ in range(n_lines)]
    
    for i, (x, y) in enumerate(points):
        for j, (A, B, C) in enumerate(lines):
            if A * x + B * y + C == 0:
                counts[j] += 1
                point_sets[j].add(i)
                
    return counts, point_sets

def _get_all_lines_through_pairs(points):
    """
    Generate all unique lines through pairs of points.
    Returns a dictionary: normalized_line -> (A, B, C)
    """
    n_pts = len(points)
    line_map = {}
    
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            if x1 == x2:
                A, B, C = 1, 0, -x1
            elif y1 == y2:
                A, B, C = 0, 1, -y1
            else:
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                # Simplify by dividing by gcd of A, B, C? 
                # The normalization function handles it, but let's keep numbers small.
                # Actually, normalization is required for the key.
            
            nline = _line_key(A, B, C)
            if nline not in line_map:
                line_map[nline] = (A, B, C)
                
    return line_map

def _count_incidence(points, lines):
    """Calculate total incidences for given points and lines."""
    I = 0
    for (A, B, C) in lines:
        for (x, y) in points:
            if A * x + B * y + C == 0:
                I += 1
    return I

def _optimize_lines_for_points(points, deadline):
    """
    Given fixed points, find the best 27 lines.
    Strategy: 
    1. Generate all lines through pairs.
    2. Score each line by number of points it passes through.
    3. Greedily select top 27.
    4. Local search: try swapping a selected line with a non-selected one if it improves total incidence.
    """
    line_map = _get_all_lines_through_pairs(points)
    if not line_map:
        return [(1, 0, 0), (0, 1, 0)] * 13 # Fallback
        
    # Score lines
    line_scores = {}
    for nline, (A, B, C) in line_map.items():
        count = 0
        for (x, y) in points:
            if A * x + B * y + C == 0:
                count += 1
        line_scores[nline] = count
        
    # Sort by score
    sorted_lines = sorted(line_scores.items(), key=lambda x: (-x[1], x[0]))
    
    # Initial selection: Top 27
    selected_lines = [line_map[nline] for nline, _ in sorted_lines[:27]]
    selected_keys = set([line_map[nline] and _line_key(*line_map[nline]) for nline, _ in sorted_lines[:27]])
    
    current_score = _count_incidence(points, selected_lines)
    
    # Local search on lines
    # We can try swapping a low-scoring selected line with a high-scoring unselected line.
    # However, incidence is not additive because points are shared.
    # A better local search: 
    # 1. Calculate incidence contribution of each selected line.
    # 2. Find the line with lowest "marginal" contribution (hard to define simply).
    # Let's stick to a simpler heuristic: 
    # Re-evaluate total incidence after swap.
    
    # To make this fast, we can precompute point-to-line incidences.
    # But for 54 points and ~1000 lines, it's manageable.
    
    # Let's do a few iterations of swapping.
    # We need to know which lines are selected.
    # We need to know which lines are candidates for replacement.
    
    # For efficiency in the main loop, we will just use the greedy top-27.
    # The point optimization will handle the rest.
    
    return selected_lines

def _optimize_points_for_lines(points, lines, deadline):
    """
    Given fixed lines, optimize point positions to maximize incidences.
    Uses simulated annealing with fast incremental updates.
    """
    n_points = 54
    n_lines = len(lines)
    
    # Precompute line coefficients
    line_coeffs = np.array(lines, dtype=np.int64)
    
    # Current state
    current_points = points.copy()
    occupied = set(map(tuple, current_points))
    
    # Incidence matrix: point i is on line j?
    # We maintain counts per line, and a set of points per line.
    line_point_sets = [set() for _ in range(n_lines)]
    for i, (x, y) in enumerate(current_points):
        for j in range(n_lines):
            A, B, C = line_coeffs[j]
            if A * x + B * y + C == 0:
                line_point_sets[j].add(i)
                
    current_score = sum(len(s) for s in line_point_sets)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() < deadline and T > T_min:
        # Pick a random point to move
        idx = random.randint(0, n_points - 1)
        orig_x, orig_y = current_points[idx]
        
        # Generate move
        # Strategy: 
        # 1. Small random perturbation (explore)
        # 2. Move to a random point on one of the lines (exploit)
        
        if random.random() < 0.7:
            # Move to a point on one of the lines
            line_idx = random.randint(0, n_lines - 1)
            A, B, C = line_coeffs[line_idx]
            
            # Find a random integer point on this line near the current point
            # A*x + B*y + C = 0
            # We want x' close to orig_x, y' close to orig_y.
            # Let's sample x' in a range around orig_x and solve for y'
            # If B != 0, y' = (-A*x' - C) / B. Must be integer.
            # If B == 0, x' = -C/A. Must be integer and unique.
            
            found_move = False
            # Try a few random x values
            for _ in range(10):
                dx = random.randint(-5, 5)
                new_x = orig_x + dx
                if B != 0:
                    num = -A * new_x - C
                    if num % B == 0:
                        new_y = num // B
                        if (new_x, new_y) not in occupied:
                            found_move = True
                            break
                else:
                    # Vertical line
                    if A != 0 and C % A == 0:
                        new_x = -C // A
                        # y can be anything, pick near orig_y
                        new_y = orig_y + random.randint(-2, 2)
                        if (new_x, new_y) not in occupied:
                            found_move = True
                            break
            if found_move:
                pass
            else:
                # Fallback to random perturbation
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                new_x = orig_x + dx
                new_y = orig_y + dy
                if (new_x, new_y) in occupied:
                    T *= cooling_rate
                    continue
        else:
            # Random perturbation
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            new_x = orig_x + dx
            new_y = orig_y + dy
            if (new_x, new_y) in occupied:
                T *= cooling_rate
                continue
        
        if new_x == orig_x and new_y == orig_y:
            continue
            
        # Calculate delta score
        # Remove point idx from all lines it's on
        delta = 0
        removed_lines = []
        for j in range(n_lines):
            if idx in line_point_sets[j]:
                line_point_sets[j].remove(idx)
                delta -= 1
                removed_lines.append(j)
        
        # Add point idx to lines it's on at new position
        added_lines = []
        for j in range(n_lines):
            A, B, C = line_coeffs[j]
            if A * new_x + B * new_y + C == 0:
                line_point_sets[j].add(idx)
                delta += 1
                added_lines.append(j)
                
        new_score = current_score + delta
        
        # Acceptance
        if delta >= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_points[idx] = [new_x, new_y]
            occupied.remove((orig_x, orig_y))
            occupied.add((new_x, new_y))
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Revert
            # Re-add to removed lines
            for j in removed_lines:
                line_point_sets[j].add(idx)
            # Remove from added lines
            for j in added_lines:
                line_point_sets[j].remove(idx)
        
        T *= cooling_rate
        
    return best_points, best_score

def _co_optimize(points, deadline):
    """
    Alternate between optimizing lines for fixed points and points for fixed lines.
    """
    current_points = points.copy()
    
    # Initial lines
    current_lines = _optimize_lines_for_points(current_points, deadline)
    
    best_points = current_points.copy()
    best_lines = current_lines
    best_score = _count_incidence(current_points, current_lines)
    
    start_time = time.time()
    time_per_phase = 50 # seconds
    
    while time.time() < deadline:
        phase_deadline = time.time() + min(time_per_phase, deadline - time.time())
        
        # Optimize points for current lines
        new_points, score = _optimize_points_for_lines(current_points, current_lines, phase_deadline)
        
        if score > best_score:
            best_points = new_points.copy()
            best_lines = current_lines.copy()
            best_score = score
            current_points = new_points.copy()
            
        if time.time() >= deadline:
            break
            
        # Optimize lines for current points
        phase_deadline = time.time() + min(time_per_phase, deadline - time.time())
        new_lines = _optimize_lines_for_points(current_points, phase_deadline)
        
        score = _count_incidence(current_points, new_lines)
        
        if score > best_score:
            best_score = score
            best_lines = new_lines.copy()
            best_points = current_points.copy()
            current_lines = new_lines.copy()
            
    return best_points, best_lines, best_score

def _initial_grid_config():
    """Create initial 6x9 grid configuration."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64).copy()
    return points

def szemeredi_trotter_incidences():
    """
    Returns 54 points and 27 lines maximizing incidences.
    Uses co-optimization of points and lines.
    """
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    random.seed(42)
    
    # Start with a grid
    points = _initial_grid_config()
    
    best_points, best_lines, best_score = _co_optimize(points, deadline)
    
    if best_points is None:
        # Fallback
        points = _initial_grid_config()
        lines = _optimize_lines_for_points(points, deadline)
        return points, np.array(lines, dtype=np.int64)
        
    return best_points, np.array(best_lines, dtype=np.int64)
