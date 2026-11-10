# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from math import gcd

def _normalize_line(A, B, C):
    """
    Normalize line coefficients A, B, C to a canonical form.
    Returns (A, B, C) or None if invalid.
    """
    if A == 0 and B == 0:
        return None
    
    # Determine GCD
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _line_from_points(p1, p2):
    """
    Calculate line coefficients from two points (x1, y1), (x2, y2).
    """
    x1, y1 = p1
    x2, y2 = p2
    
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    
    return _normalize_line(A, B, C)

def _count_incidences(points, lines):
    """
    Count incidences exactly.
    points: (N, 2)
    lines: (M, 3)
    """
    count = 0
    for p in points:
        x, y = p
        for l in lines:
            A, B, C = l
            if A*x + B*y + C == 0:
                count += 1
    return count

def _line_key(line):
    """
    Get a hashable key for a line to check for duplicates.
    """
    if line is None:
        return None
    A, B, C = line
    return (A, B, C)

def szemeredi_trotter_incidences():
    """
    Uses a Hill Climbing approach to maximize incidences.
    Starts from a 6x9 grid configuration.
    """
    # 1. Initialize with the known good grid configuration
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines_list = []
    for i in range(1, 7):
        lines_list.append((1, 0, -i))
    for j in range(1, 10):
        lines_list.append((0, 1, -j))
    for c in [0, 1, 2, 3]:
        lines_list.append((-1, 1, -c))
    for c in [7, 8, 9, 10]:
        lines_list.append((1, 1, -c))
    for c in [-1, 4]:
        lines_list.append((-1, 1, -c))
    for c in [6, 11]:
        lines_list.append((1, 1, -c))
        
    lines = np.array(lines_list, dtype=np.int64)
    
    # Convert to list of tuples for easier manipulation
    lines_tuple_list = [tuple(l) for l in lines]
    
    current_score = _count_incidences(points, lines_tuple_list)
    
    # 2. Hill Climbing
    start_time = time.time()
    max_time = 50.0 # Leave buffer for evaluation overhead
    random.seed(42)
    
    point_set = set(map(tuple, points))
    
    while time.time() - start_time < max_time:
        move_type = random.choice(['point', 'line'])
        
        if move_type == 'point':
            # Pick a random point to move
            idx = random.randint(0, 53)
            p = points[idx]
            # Try small moves
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            
            new_p = (p[0] + dx, p[1] + dy)
            
            # Check if new point is valid (integer, not overlapping)
            if new_p in point_set:
                continue
            
            # Update point set
            point_set.remove(tuple(p))
            point_set.add(new_p)
            
            # Update points array
            points[idx] = new_p
            
            new_score = _count_incidences(points, lines_tuple_list)
            
            if new_score > current_score:
                current_score = new_score
            else:
                # Revert
                points[idx] = p
                point_set.remove(new_p)
                point_set.add(tuple(p))
                
        else:
            # Pick a random line to replace
            l_idx = random.randint(0, 26)
            old_line = lines_tuple_list[l_idx]
            
            # Pick two random points to define a new line
            p1_idx = random.randint(0, 53)
            p2_idx = random.randint(0, 53)
            if p1_idx == p2_idx:
                continue
            
            p1 = points[p1_idx]
            p2 = points[p2_idx]
            
            new_line = _line_from_points(p1, p2)
            
            if new_line is None:
                continue
            
            # Check if new line is already in the list (other than the one being replaced)
            # We need to be careful with duplicates
            line_set = set(lines_tuple_list)
            line_set.remove(old_line)
            
            if new_line in line_set:
                continue
            
            # Update lines list
            lines_tuple_list[l_idx] = new_line
            
            new_score = _count_incidences(points, lines_tuple_list)
            
            if new_score > current_score:
                current_score = new_score
            else:
                # Revert
                lines_tuple_list[l_idx] = old_line
                
    # Final conversion to numpy arrays
    final_points = points.astype(np.int64)
    final_lines = np.array(lines_tuple_list, dtype=np.int64)
    
    return final_points, final_lines

# EVOLVE-BLOCK-END
