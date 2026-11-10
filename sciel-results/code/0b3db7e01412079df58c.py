import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def generate_grid_configuration(k=3):
    """
    Generate a base configuration using a grid structure.
    Points: (x, y) for x in [1..k], y in [1..2k^2] -> 3 * 18 = 54 points.
    Lines: 
      1. Vertical lines x = i (k lines)
      2. Horizontal-ish lines y = ax + b. 
         In the current seed: a in [1..k], b in [1..k^2].
         Incidence condition: y = ax + b.
         For a fixed x, y ranges over [1..2k^2].
         For a fixed a, b ranges over [1..k^2].
         y - ax = b. Since b is integer, we need y - ax to be in [1..k^2].
    """
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    # Vertical lines: x = i => 1*x + 0*y - i = 0
    for i in range(1, k + 1):
        lines.append((1, 0, -i))
        
    # Slanted lines: a*x - y + b = 0 => y = ax + b
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines.append((a, -1, b))
            
    lines = np.array(lines, dtype=np.int64)
    return points, lines

def local_search_fast(points, lines, time_limit=500):
    """
    Fast local search using incremental incidence updates.
    Instead of recomputing the full 54x27 matrix every step,
    we track the incidence count for each point and each line.
    
    When moving a point p_i to p_i':
    1. Remove incidences of p_i with all lines.
    2. Add incidences of p_i' with all lines.
    3. Update total I.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    P = points.astype(np.int64).copy()
    L = lines.astype(np.int64).copy()
    
    # Precompute incidence matrix (54, 27)
    # M[i, j] = 1 if P[i] lies on L[j]
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    # M is boolean (54, 27)
    M = (A * px + B * py + C) == 0
    
    # Point incidences: number of lines each point is on
    point_counts = M.sum(axis=1)
    # Line incidences: number of points each line passes through
    line_counts = M.sum(axis=0)
    
    current_I = int(M.sum())
    best_I = current_I
    best_P = P.copy()
    best_L = L.copy()
    
    # Candidate points for moves: a larger grid
    box_range = 15
    xs = np.arange(-box_range, box_range + 1)
    ys = np.arange(-box_range, box_range + 1)
    all_candidates = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1, 2)
    
    # Remove current points from candidates? No, we just check uniqueness.
    # For speed, we maintain a set of current point coordinates for O(1) lookup
    current_point_set = set(map(tuple, P))
    
    T = 10.0
    T_min = 0.5
    alpha = 0.999
    
    while time.time() < deadline:
        idx = random.randint(0, 53)
        old_point = P[idx].copy()
        
        # Pick a new candidate point
        # Bias towards points that might have high incidence?
        # Random selection from a local neighborhood is often better for escaping local minima
        # Let's try a mix: 50% random from grid, 50% random neighbor
        if random.random() < 0.5:
            new_point = all_candidates[random.randint(0, len(all_candidates) - 1)].copy()
        else:
            new_point = old_point + np.array([random.randint(-2, 2), random.randint(-2, 2)])
            if new_point[0] < -box_range: new_point[0] = -box_range
            if new_point[0] > box_range: new_point[0] = box_range
            if new_point[1] < -box_range: new_point[1] = -box_range
            if new_point[1] > box_range: new_point[1] = box_range
            
        new_point_tuple = tuple(new_point)
        if new_point_tuple in current_point_set:
            continue
            
        # Calculate delta I
        # Remove old point incidences
        old_inc = M[idx, :]  # (27,) boolean
        # Add new point incidences
        # Check new point against all lines
        vals = A * new_point[0] + B * new_point[1] + C # (1, 27)
        new_inc = (vals == 0).flatten() # (27,)
        
        delta = int(new_inc.sum()) - int(old_inc.sum())
        
        # Acceptance probability
        if delta > 0 or random.random() < np.exp(delta / T):
            # Update state
            M[idx, :] = new_inc
            # Update line counts
            # line_counts[j] += new_inc[j] - old_inc[j]
            line_counts += new_inc.astype(np.int64) - old_inc.astype(np.int64)
            
            # Update point set
            current_point_set.remove(tuple(old_point))
            current_point_set.add(new_point_tuple)
            
            P[idx] = new_point
            current_I += delta
            
            if current_I > best_I:
                best_I = current_I
                best_P = P.copy()
                best_L = L.copy()
        else:
            # Reject (no state change needed since we didn't modify P/M yet)
            pass
            
        T *= alpha
        if T < T_min:
            T = T_min
            # Occasionally reheat to escape local minima
            if random.random() < 0.01:
                T = 10.0

    return best_P, best_L, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 620 # 10s safety margin
    
    # Strategy 1: Deterministic Grid Construction (Fast Baseline)
    points_grid, lines_grid = generate_grid_configuration(k=3)
    I_grid = count_incidences(points_grid, lines_grid)
    
    best_points = points_grid
    best_lines = lines_grid
    best_I = I_grid
    
    # Strategy 2: Fast Local Search from Grid
    if time.time() < deadline - 10:
        time_limit_ls = min(500, deadline - time.time() - 10)
        points_ls, lines_ls, I_ls = local_search_fast(points_grid, lines_grid, time_limit=time_limit_ls)
        
        if I_ls > best_I:
            best_I = I_ls
            best_points = points_ls
            best_lines = lines_ls
            
    # Strategy 3: Try different grid parameters if time permits
    if time.time() < deadline - 60:
        # Try k=2? 2 * 8 = 16 points. Too few.
        # Try a different structure?
        # For now, stick with the best found.
        pass
        
    # Final Validation
    # Ensure points are unique
    unique_points, indices = np.unique(best_points, axis=0, return_index=True)
    if len(unique_points) != 54:
        # Fallback
        best_points, best_lines = generate_grid_configuration(k=3)
    else:
        best_points = unique_points[indices]
        
    # Ensure lines are unique and valid
    unique_lines = []
    for line in best_lines:
        n_line = normalize_line(line[0], line[1], line[2])
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        best_points, best_lines = generate_grid_configuration(k=3)
    else:
        best_lines = np.array(unique_lines[:27], dtype=np.int64)
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
