# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calculate_incidence(points, lines):
    """
    Calculate the total number of incidences.
    points: (N, 2) int array
    lines: (M, 3) int array
    Returns: int
    """
    # Vectorized check
    # points[:, 0] is x, points[:, 1] is y
    # lines[:, 0] is A, lines[:, 1] is B, lines[:, 2] is C
    # We want A*x + B*y + C == 0
    
    # Broadcasting: 
    # x: (N, 1), y: (N, 1)
    # A: (1, M), B: (1, M), C: (1, M)
    # Result: (N, M)
    
    x = points[:, 0:1].astype(np.int64)
    y = points[:, 1:2].astype(np.int64)
    A = lines[:, 0:1].T.astype(np.int64)
    B = lines[:, 1:2].T.astype(np.int64)
    C = lines[:, 2:3].T.astype(np.int64)
    
    vals = A * x + B * y + C
    mask = (vals == 0)
    return np.sum(mask)

def _normalize_line(A, B, C):
    """
    Normalize line coefficients so that gcd(|A|, |B|, |C|) = 1
    and the first non-zero coefficient is positive.
    """
    g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0:
        if B < 0:
            B = -B
            C = -C
        elif B == 0:
            if C < 0:
                C = -C
    
    return (A, B, C)

def _line_from_points(p1, p2):
    """
    Get line coefficients (A, B, C) for line passing through p1, p2.
    p1, p2 are (x, y) tuples.
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)

def _get_lines_from_points(points):
    """
    Generate a pool of candidate lines from the current points.
    """
    lines_set = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            l = _line_from_points(points[i], points[j])
            lines_set.add(l)
    return list(lines_set)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 580  # Safety margin
    
    # 1. Initial Grid: 9x6
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Initial Lines (Optimal for grid)
    lines_list = []
    for j in range(1, 7): lines_list.append((0, 1, -j))
    for i in range(1, 10): lines_list.append((1, 0, -i))
    for c in [0, 1, 2, 3]: lines_list.append((1, -1, -c))
    for c in [7, 8, 9, 10]: lines_list.append((1, 1, -c))
    for c in [-1, -3]: lines_list.append((1, -2, -c))
    for c in [11, 13]: lines_list.append((1, 2, -c))
    
    lines = np.array(lines_list, dtype=np.int64)
    
    current_score = _calculate_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    # Precompute line incidences for current state to speed up swaps?
    # For now, just use full recalculation as it's fast enough for 54x27.
    
    # Local Search
    # Strategy: 
    # 1. Try moving a point to a random nearby location (or swap with another point)
    # 2. Try replacing a line with a line defined by two random points
    
    n_points = 54
    n_lines = 27
    
    # Temperature for simulated annealing
    T = 5.0
    min_T = 0.1
    
    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    step = 0
    while time.time() - start_time < deadline:
        # Annealing
        T = max(min_T, T * 0.999)
        
        # Choose move type
        move_type = random.random()
        
        # Copy current state
        new_points = points.copy()
        new_lines = lines.copy()
        
        if move_type < 0.5:
            # Move a point
            idx = random.randint(0, n_points - 1)
            # Try moving to a random position in a small neighborhood
            # Or swap with another point
            if random.random() < 0.5:
                # Swap
                other_idx = random.randint(0, n_points - 1)
                if other_idx != idx:
                    new_points[idx], new_points[other_idx] = new_points[other_idx].copy(), new_points[idx].copy()
            else:
                # Small shift
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                new_point = points[idx] + np.array([dx, dy])
                # Check if new point is unique (not already in points)
                # For efficiency, we might skip strict uniqueness check during search 
                # and rely on the final validation, but duplicates are invalid.
                # We must ensure no duplicates.
                # Simple check:
                is_dup = False
                for p in points:
                    if np.array_equal(p, new_point) and not np.array_equal(p, points[idx]):
                         is_dup = True
                         break
                if not is_dup:
                    new_points[idx] = new_point
        else:
            # Replace a line
            idx = random.randint(0, n_lines - 1)
            # Pick two random points to define a new line
            i1 = random.randint(0, n_points - 1)
            i2 = random.randint(0, n_points - 1)
            if i1 == i2:
                continue
            new_line = _line_from_points(points[i1], points[i2])
            new_lines[idx] = new_line
            # Ensure no duplicate lines
            # Check if new_line is already in new_lines (excluding idx)
            is_dup_line = False
            for j in range(n_lines):
                if j != idx and np.array_equal(new_lines[j], new_line):
                    is_dup_line = True
                    break
            if is_dup_line:
                continue
        
        # Calculate new score
        new_score = _calculate_incidence(new_points, new_lines)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta >= 0 or random.random() < np.exp(delta / T):
            points = new_points
            lines = new_lines
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
        
        step += 1
        # Throttle if too fast? No, 600s is plenty.
        
    # Final validation and cleanup
    # Ensure points are unique and lines are valid
    # The search might produce invalid states (duplicates).
    # We return the best valid state found.
    
    # Check best_points for duplicates
    unique_points = np.unique(best_points, axis=0)
    if len(unique_points) != 54:
        # Fallback to initial grid if best is invalid
        return best_points if len(np.unique(best_points, axis=0))==54 else points, best_lines
    
    # Check best_lines for duplicates and validity
    # Normalize all lines
    norm_lines = []
    for l in best_lines:
        norm_lines.append(_normalize_line(l[0], l[1], l[2]))
    
    unique_lines = []
    seen = set()
    valid = True
    for l in norm_lines:
        if l in seen:
            valid = False
            break
        seen.add(l)
        unique_lines.append(l)
        
    if not valid or len(unique_lines) != 27:
        # Fallback
        return points, lines

    return best_points, np.array(unique_lines, dtype=np.int64)

# EVOLVE-BLOCK-END
