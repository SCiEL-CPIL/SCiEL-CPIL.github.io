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

def _get_best_lines(points, num_lines=27):
    """
    Select the best num_lines from all lines through pairs of points.
    Returns list of (A, B, C) tuples.
    """
    n_pts = len(points)
    line_map = {}     # normalized line -> (A, B, C)
    
    # Generate all unique lines defined by pairs of points
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
    
    # Sort by count descending, break ties by line tuple for determinism
    sorted_lines = sorted(line_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Pick top num_lines
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

def szemeredi_trotter_incidences():
    """
    Returns 54 points and 27 lines maximizing incidences.
    Uses Simulated Annealing on both points and lines to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initialize with 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    current_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64).copy()
    
    # Get initial best lines and incidence
    current_lines = _get_best_lines(current_points, 27)
    current_score = _calculate_incidence(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines
    best_score = current_score
    
    random.seed(42)
    
    # Simulated Annealing Parameters
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random move type
        move_type = random.random()
        
        if move_type < 0.8:
            # Point move: Move a random point to a nearby integer coordinate
            idx = random.randint(0, 53)
            orig_x, orig_y = current_points[idx]
            
            # Generate a random small perturbation
            # Bias towards small moves to maintain structure
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            
            if dx == 0 and dy == 0:
                continue
                
            new_x = orig_x + dx
            new_y = orig_y + dy
            
            # Check if new position is already occupied by another point
            occupied = False
            for k in range(54):
                if k == idx:
                    continue
                if current_points[k][0] == new_x and current_points[k][1] == new_y:
                    occupied = True
                    break
            
            if occupied:
                continue
            
            # Apply move
            current_points[idx] = [new_x, new_y]
            
            # Recalculate best lines and score
            # Optimization: Instead of recalculating ALL lines, we can just
            # check the current lines + the new lines defined by the moved point.
            # However, for simplicity and robustness in this budget, we use the
            # full recalculation but optimize the line selection.
            
            # Fast line update:
            # 1. Keep existing lines that are still valid (pass through >= 2 points)
            # 2. Add new lines formed by the moved point and others
            # 3. Select top 27
            
            # For now, using _get_best_lines is O(N^2) which is ~2900 ops, 
            # and incidence is O(N*L) ~ 1450 ops. Total ~4000 ops per iteration.
            # 580s / 4000 ops is plenty of headroom for Python if optimized.
            
            new_lines = _get_best_lines(current_points, 27)
            new_score = _calculate_incidence(current_points, new_lines)
            
            delta = new_score - current_score
            
            # Accept or reject
            if delta > 0:
                current_score = new_score
                current_lines = new_lines
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines
            else:
                # Reject with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    current_score = new_score
                    current_lines = new_lines
                else:
                    # Revert move
                    current_points[idx] = [orig_x, orig_y]
                    
        else:
            # Line move: Swap a current line with a candidate line
            # This allows escaping the "greedy best lines" trap
            if len(current_lines) < 27:
                continue
                
            # Pick a line to remove
            line_idx = random.randint(0, 26)
            orig_line = current_lines[line_idx]
            
            # Generate a candidate line to add
            # Pick two random points
            p1_idx = random.randint(0, 53)
            p2_idx = random.randint(0, 53)
            if p1_idx == p2_idx:
                continue
                
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            
            x1, y1 = p1
            x2, y2 = p2
            
            if x1 == x2:
                cand_line = (1, 0, -x1)
            elif y1 == y2:
                cand_line = (0, 1, -y1)
            else:
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                cand_line = (A, B, C)
            
            cand_norm = _normalize_line(*cand_line)
            orig_norm = _normalize_line(*orig_line)
            
            if cand_norm == orig_norm:
                continue
                
            # Check if cand_line is already in current_lines (to avoid duplicates)
            is_dup = False
            for i, l in enumerate(current_lines):
                if i == line_idx:
                    continue
                if _normalize_line(*l) == cand_norm:
                    is_dup = True
                    break
            
            if is_dup:
                continue
                
            # Apply swap
            current_lines[line_idx] = cand_line
            
            new_score = _calculate_incidence(current_points, current_lines)
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            else:
                if random.random() < math.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert swap
                    current_lines[line_idx] = orig_line
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_points, np.array(best_lines, dtype=np.int64)
