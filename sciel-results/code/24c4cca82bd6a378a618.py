import time
import random
import math
from itertools import combinations

def _count_lines_exactly_4(points):
    """
    Counts lines with exactly 4 points. Returns -1 if invalid (duplicates, out of bounds, or 5+ collinear).
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check distinctness and bounds
    seen = set()
    for x, y in points:
        if (x, y) in seen:
            return -1
        seen.add((x, y))
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
            
    lines_4 = 0
    # Use a set of line signatures to avoid double counting
    # A line is defined by its set of point indices
    line_sets = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Find all points on this line
            indices = {i, j}
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    indices.add(k)
            
            if len(indices) >= 5:
                return -1
                
            if len(indices) == 4:
                # Canonicalize the set of indices
                key = tuple(sorted(indices))
                if key not in line_sets:
                    line_sets.add(key)
                    lines_4 += 1
                    
    return lines_4

def _evaluate(points):
    """
    Returns a score for the configuration.
    Higher is better.
    -1000 if invalid.
    """
    n = len(points)
    if n != 16:
        return -1000
    
    seen = set()
    for x, y in points:
        if (x, y) in seen:
            return -1000
        seen.add((x, y))
        if abs(x) > 1000 or abs(y) > 1000:
            return -1000
            
    lines_4 = 0
    lines_5_plus = 0
    line_sets = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            indices = {i, j}
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    indices.add(k)
            
            if len(indices) >= 5:
                lines_5_plus += 1
            elif len(indices) == 4:
                key = tuple(sorted(indices))
                if key not in line_sets:
                    line_sets.add(key)
                    lines_4 += 1
                    
    # Penalty for 5+ lines is severe
    return lines_4 - 100 * lines_5_plus

def construct():
    start_time = time.time()
    deadline = start_time + 120 # Safety margin
    
    # Define a candidate grid. 
    # The 4x4 grid maps to indices 0-3 in x and y.
    # Let's use a 7x7 grid centered at 0, so coords -3 to 3.
    # But we can use any integers within bounds.
    # Let's stick to small integers to keep lines "dense" and likely to form 4-point lines.
    # Grid size 7x7: coords in [-3, 3]
    grid_size = 7
    offset = grid_size // 2
    grid_points = []
    for x in range(grid_size):
        for y in range(grid_size):
            grid_points.append([x - offset, y - offset])
            
    # Total grid points
    G = len(grid_points)
    
    # Initial configuration: The 4x4 grid subset of the 7x7 grid.
    # 4x4 grid in the center of 7x7:
    # Indices for x in [1,4] and y in [1,4] of the 0-6 grid
    # x=1 -> -2, x=2 -> -1, x=3 -> 0, x=4 -> 1
    # Let's just pick 16 points from the grid.
    # The standard 4x4 grid is:
    # (-2,-2) to (1,1)
    init_indices = []
    for x in range(1, 5):
        for y in range(1, 5):
            idx = x * grid_size + y
            init_indices.append(idx)
            
    current_indices = init_indices[:]
    current_points = [grid_points[i] for i in current_indices]
    current_score = _evaluate(current_points)
    
    best_indices = current_indices[:]
    best_points = current_points[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    # Move strategies:
    # 1. Change one point index
    # 2. Change two point indices (swap or reassign)
    
    iter_count = 0
    while time.time() < deadline:
        iter_count += 1
        
        # Generate neighbor
        new_indices = current_indices[:]
        move_type = random.random()
        
        if move_type < 0.7:
            # Change one point
            idx_to_change = random.randint(0, 15)
            new_pos = random.randint(0, G - 1)
            # Ensure it's not already occupied by another point
            if new_pos not in current_indices:
                new_indices[idx_to_change] = new_pos
            else:
                # Try a few times to find a free spot
                for _ in range(5):
                    new_pos = random.randint(0, G - 1)
                    if new_pos not in current_indices:
                        new_indices[idx_to_change] = new_pos
                        break
        else:
            # Change two points
            idx1, idx2 = random.sample(range(16), 2)
            new_pos1 = random.randint(0, G - 1)
            new_pos2 = random.randint(0, G - 1)
            if new_pos1 not in current_indices and new_pos2 not in current_indices and new_pos1 != new_pos2:
                new_indices[idx1] = new_pos1
                new_indices[idx2] = new_pos2
            else:
                # Fallback to single change if collision
                idx_to_change = random.randint(0, 15)
                new_pos = random.randint(0, G - 1)
                if new_pos not in current_indices:
                    new_indices[idx_to_change] = new_pos
        
        if new_indices == current_indices:
            continue
            
        new_points = [grid_points[i] for i in new_indices]
        new_score = _evaluate(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
            current_indices = new_indices
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_indices = current_indices[:]
                best_points = current_points[:]
                
        T = max(T_min, T * cooling)
        
        # Restart temperature occasionally if stuck?
        # Or just continue.
        
    # Final validation
    final_score = _count_lines_exactly_4(best_points)
    if final_score >= 0:
        return best_points
    else:
        # Fallback to 4x4 grid
        return [[x - 2, y - 2] for x in range(4) for y in range(4)]

