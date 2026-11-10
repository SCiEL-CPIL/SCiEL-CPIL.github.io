import numpy as np
import time
from math import gcd
from itertools import combinations

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

def get_line_from_points(p1, p2):
    """Get normalized line coefficients from two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x2
    return normalize_line(A, B, C)

def generate_initial_grid(w, h):
    """Generate a w x h grid of points centered at origin."""
    xs = np.arange(-w//2, w//2 + (w%2))
    ys = np.arange(-h//2, h//2 + (h%2))
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def generate_initial_random(n, box=10):
    """Generate n random unique points in a box."""
    points = set()
    while len(points) < n:
        points.add((np.random.randint(-box, box+1), np.random.randint(-box, box+1)))
    return np.array(list(points), dtype=np.int64)

def select_best_lines(points, k_lines):
    """
    Given a set of points, find the k_lines lines that maximize incidences.
    We consider lines defined by pairs of points.
    """
    n = len(points)
    if n < 2:
        return np.zeros((k_lines, 3), dtype=np.int64)
    
    # Count incidences for all lines defined by pairs
    # This is O(n^2 * n) which is 54^3 ~ 150k, acceptable.
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            line = get_line_from_points(points[i], points[j])
            if line is None:
                continue
            if line not in line_counts:
                line_counts[line] = 0
            # We know i and j are on the line. We need to count others.
            # For efficiency in this small case, we can just count all points for each unique line later.
            # But we can optimize: just record the line.
    
    # Now count actual incidences for each unique line
    # To speed this up, we can use the fact that we only need the top k.
    # But with 54 points, there are at most ~1400 unique lines.
    # Checking 1400 lines against 54 points is 1400*54 ~ 75k ops. Very fast.
    
    unique_lines = list(line_counts.keys())
    if not unique_lines:
        return np.zeros((k_lines, 3), dtype=np.int64)
        
    L = np.array(unique_lines, dtype=np.int64)
    P = points.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k_lines
    top_indices = np.argsort(counts)[::-1][:k_lines]
    best_lines = L[top_indices]
    
    return best_lines

def local_search(points, lines, time_limit):
    """
    Greedy local search:
    1. Try moving each point to a neighbor (dx, dy) in [-2, 2]^2.
    2. Try replacing each line with the best line from the current point set.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    current_I = count_incidences(points, lines)
    best_I = current_I
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Precompute neighbor offsets
    neighbors = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if not (dx==0 and dy==0)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Point Moves
        for i in range(len(points)):
            old_p = points[i].copy()
            best_gain = 0
            best_new_p = old_p
            
            for dx, dy in neighbors:
                new_p = old_p + np.array([dx, dy])
                
                # Check uniqueness
                if np.any(np.all(points == new_p, axis=1)):
                    continue
                
                points[i] = new_p
                new_I = count_incidences(points, lines)
                gain = new_I - current_I
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_p = new_p.copy()
                
                points[i] = old_p
            
            if best_gain > 0:
                points[i] = best_new_p
                current_I += best_gain
                improved = True
                if current_I > best_I:
                    best_I = current_I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        # 2. Line Moves
        # For each line, check if there's a better line defined by the current points
        # We can just re-select the top 27 lines from the current point set
        # and see if the total I increases.
        # This is a "global" line update.
        
        # Calculate current line incidences individually to see if swapping helps?
        # Simpler: Just try replacing the worst-performing line with the best available line not in the set.
        
        # Let's calculate the incidence count for each current line
        L = lines.astype(np.int64)
        P = points.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        line_counts = np.sum(vals == 0, axis=1)
        
        # Find the line with the minimum count
        min_idx = np.argmin(line_counts)
        min_count = line_counts[min_idx]
        
        # Find the best line from the point set that is NOT in the current lines
        all_lines = select_best_lines(points, 50) # Get top 50 candidates
        current_line_set = set(map(tuple, lines))
        
        best_candidate = None
        best_candidate_count = 0
        
        for cand_line in all_lines:
            if tuple(cand_line) in current_line_set:
                continue
            # Calculate incidence count of this candidate
            # We can reuse the vals calculation if we were careful, but it's small.
            # A*x + B*y + C == 0
            cnt = np.sum(cand_line[0]*points[:,0] + cand_line[1]*points[:,1] + cand_line[2] == 0)
            if cnt > best_candidate_count:
                best_candidate_count = cnt
                best_candidate = cand_line
                
        if best_candidate is not None and best_candidate_count > min_count:
            lines[min_idx] = best_candidate
            current_I += (best_candidate_count - min_count)
            improved = True
            if current_I > best_I:
                best_I = current_I
                best_points = points.copy()
                best_lines = lines.copy()
                
    return best_points, best_lines, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 620 # Safety margin
    
    best_points_global = None
    best_lines_global = None
    best_I_global = -1
    
    # Strategy 1: Standard Grid Baseline
    # 54 points. 6x9 grid?
    points_grid = generate_initial_grid(6, 9)
    lines_grid = select_best_lines(points_grid, 27)
    I_grid = count_incidences(points_grid, lines_grid)
    
    best_I_global = I_grid
    best_points_global = points_grid
    best_lines_global = lines_grid
    
    # Strategy 2: Local Search from Grid
    if time.time() < deadline - 10:
        p, l, I = local_search(points_grid.copy(), lines_grid.copy(), time_limit=60)
        if I > best_I_global:
            best_I_global = I
            best_points_global = p
            best_lines_global = l
            
    # Strategy 3: Random Start + Local Search
    if time.time() < deadline - 10:
        points_rand = generate_initial_random(54, box=15)
        lines_rand = select_best_lines(points_rand, 27)
        I_rand = count_incidences(points_rand, lines_rand)
        
        if I_rand > best_I_global:
            best_I_global = I_rand
            best_points_global = points_rand
            best_lines_global = lines_rand
            
        if time.time() < deadline - 10:
            p, l, I = local_search(points_rand.copy(), lines_rand.copy(), time_limit=60)
            if I > best_I_global:
                best_I_global = I
                best_points_global = p
                best_lines_global = l
                
    # Strategy 4: "Pencil" Structure
    # Points on a line, plus some off it.
    # Or points forming a triangle, lines through vertices.
    # Let's try a 3x18 grid (very thin)
    if time.time() < deadline - 10:
        points_thin = generate_initial_grid(3, 18)
        lines_thin = select_best_lines(points_thin, 27)
        I_thin = count_incidences(points_thin, lines_thin)
        
        if I_thin > best_I_global:
            best_I_global = I_thin
            best_points_global = points_thin
            best_lines_global = lines_thin
            
        if time.time() < deadline - 10:
            p, l, I = local_search(points_thin.copy(), lines_thin.copy(), time_limit=60)
            if I > best_I_global:
                best_I_global = I
                best_points_global = p
                best_lines_global = l
                
    # Strategy 5: "Parabola" / Algebraic Curve
    # Points (i, i^2) for i in range.
    if time.time() < deadline - 10:
        pts = []
        i = 0
        while len(pts) < 54:
            pts.append((i, i*i))
            i += 1
        points_curve = np.array(pts, dtype=np.int64)
        lines_curve = select_best_lines(points_curve, 27)
        I_curve = count_incidences(points_curve, lines_curve)
        
        if I_curve > best_I_global:
            best_I_global = I_curve
            best_points_global = points_curve
            best_lines_global = lines_curve
            
        if time.time() < deadline - 10:
            p, l, I = local_search(points_curve.copy(), lines_curve.copy(), time_limit=60)
            if I > best_I_global:
                best_I_global = I
                best_points_global = p
                best_lines_global = l

    # Final Validation and Cleanup
    if best_points_global is None:
        return generate_initial_grid(6, 9), select_best_lines(generate_initial_grid(6, 9), 27)

    # Ensure points are unique
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        # Fallback
        return generate_initial_grid(6, 9), select_best_lines(generate_initial_grid(6, 9), 27)
    else:
        best_points_global = unique_points[indices]

    # Ensure lines are unique and valid
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(line[0], line[1], line[2])
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        # Fallback
        return generate_initial_grid(6, 9), select_best_lines(generate_initial_grid(6, 9), 27)
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
        
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
