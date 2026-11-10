# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _time_left(start_time, deadline=630, safety_margin=10):
    return time.time() - start_time < deadline - safety_margin

def _canon_line(A, B, C):
    """Canonicalize line (A, B, C) -> (A', B', C') with gcd 1 and first non-zero positive."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    
    # Find GCD of absolute values
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    g = 0
    for x in [abs(A), abs(B), abs(C)]:
        g = gcd(g, x)
    
    if g > 0:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero is positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        B, C = -B, -C
        
    return (A, B, C)

def _line_from_two_points(p1, p2):
    """Return canonical line (A, B, C) passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _canon_line(A, B, C)

def _incidence_count(points, lines):
    """Count incidences. points: (n, 2) int, lines: (m, 3) int."""
    # A*x + B*y + C == 0
    # points: (54, 1, 2)
    # lines: (1, 27, 3)
    # result: (54, 27)
    val = points[:, 0:1, 0] * lines[0:1, :, 0] + \
          points[:, 0:1, 1] * lines[0:1, :, 1] + \
          lines[0:1, :, 2]
    return np.sum(val == 0)

def _get_lines_from_points(points):
    """
    Generate lines from points.
    For n=54, total pairs is 54*53/2 = 1431.
    We return a list of unique canonical lines.
    """
    lines_set = set()
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            line = _line_from_two_points(points[i], points[j])
            lines_set.add(line)
    
    # We need exactly 27 lines.
    # Which ones to pick? The ones that cover the most points.
    lines_list = list(lines_set)
    
    # Score each line by number of points on it
    scored_lines = []
    for l in lines_list:
        A, B, C = l
        count = 0
        for p in points:
            if A*p[0] + B*p[1] + C == 0:
                count += 1
        scored_lines.append((count, l))
        
    # Sort by count descending
    scored_lines.sort(key=lambda x: -x[0])
    
    # Pick top 27
    selected = [l for _, l in scored_lines[:27]]
    
    # Convert to array
    lines_arr = np.array(selected, dtype=np.int64)
    
    return lines_arr

def _local_search(points, lines, start_time, deadline=630, safety_margin=10):
    """
    Local search: move points to maximize incidences with the fixed set of lines.
    We allow points to move to any integer coordinate in a local neighborhood.
    """
    current_score = _incidence_count(points, lines)
    best_points = points.copy()
    best_score = current_score
    
    n_points = len(points)
    n_lines = len(lines)
    
    # Precompute line coefficients for fast checking
    # lines: (27, 3)
    
    while _time_left(start_time, deadline, safety_margin):
        improved = False
        # Try moving each point
        for i in range(n_points):
            if not _time_left(start_time, deadline, safety_margin):
                break
            x, y = best_points[i]
            
            # Check neighbors in a small radius
            # Radius 2 or 3 is usually enough for local optima in these problems
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    new_point = np.array([new_x, new_y], dtype=np.int64)
                    
                    # Check validity: no duplicate points
                    # Simple check: is new_point in best_points?
                    # We can do this faster with a set of tuples
                    # But for 54 points, even O(N) check is fine.
                    # Let's use a set for O(1) lookup.
                    # Actually, let's just check if any other point is at new_x, new_y.
                    # Since N is small, we can just compute incidence delta.
                    
                    # Calculate delta in incidences for point i
                    # Old incidences for point i:
                    old_val = best_points[i, 0] * lines[:, 0] + best_points[i, 1] * lines[:, 1] + lines[:, 2]
                    old_inc = np.sum(old_val == 0)
                    
                    # New incidences for point i:
                    new_val = new_x * lines[:, 0] + new_y * lines[:, 1] + lines[:, 2]
                    new_inc = np.sum(new_val == 0)
                    
                    delta = new_inc - old_inc
                    
                    if delta > 0:
                        # Check if point is already occupied
                        occupied = False
                        for j in range(n_points):
                            if j != i and best_points[j, 0] == new_x and best_points[j, 1] == new_y:
                                occupied = True
                                break
                        
                        if not occupied:
                            best_points[i] = new_point
                            current_score += delta
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points_copy = best_points.copy()
        
        if not improved:
            break
            
    return best_points, best_score

def _construct_initial_pencil():
    """
    Construct a pencil-like structure.
    Points: K_{6,9} bipartite graph vertices.
    Let U = {(0, 0), ..., (0, 5)} (6 points on x=0)
    Let V = {(1, 0), ..., (1, 8)} (9 points on x=1)
    Total 15 points? No, we need 54.
    
    Better Pencil:
    Take 6 points on x=0: P_i = (0, i) for i in 0..5
    Take 9 points on x=1: Q_j = (1, j) for j in 0..8
    This gives 15 points.
    Lines connecting P_i and Q_j.
    There are 6*9 = 54 lines.
    But we only have 27 lines.
    
    Let's use a different construction.
    Grid 6x9 = 54 points.
    Points: (i, j) for i in 0..5, j in 0..8.
    
    Lines:
    Rows: y = j (9 lines). Each has 6 points. Total 54 incidences.
    Columns: x = i (6 lines). Each has 9 points. Total 54 incidences.
    Total 15 lines.
    We need 27 lines.
    Add diagonals?
    Slope 1: x - y = c.
    Range of x-y: -8 to 5. 14 diagonals.
    Slope -1: x + y = c.
    Range of x+y: 0 to 13. 14 diagonals.
    
    If we pick 12 more lines, we can get more incidences.
    The grid is a good starting point.
    """
    # 6x9 Grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate candidate lines
    # Rows
    rows = []
    for j in range(9):
        # y - j = 0 -> 0*x + 1*y - j = 0
        rows.append((0, 1, -j))
    # Columns
    for i in range(6):
        # x - i = 0 -> 1*x + 0*y - i = 0
        rows.append((1, 0, -i))
        
    # Diagonals slope 1 (x - y = c)
    # x - y = c
    # min c: 0 - 8 = -8
    # max c: 5 - 0 = 5
    for c in range(-8, 6):
        # 1*x + (-1)*y - c = 0
        rows.append((1, -1, -c))
        
    # Diagonals slope -1 (x + y = c)
    # x + y = c
    # min c: 0
    # max c: 5 + 8 = 13
    for c in range(0, 14):
        # 1*x + 1*y - c = 0
        rows.append((1, 1, -c))
        
    # We have 9+6+14+14 = 43 candidate lines.
    # We need to pick 27 that maximize incidences.
    # Actually, we can just pick the first 27? No, let's pick the ones with most points.
    
    lines_set = set()
    for l in rows:
        lines_set.add(_canon_line(*l))
        
    lines_list = list(lines_set)
    
    # Score lines
    scored = []
    for l in lines_list:
        A, B, C = l
        count = 0
        for p in points:
            if A*p[0] + B*p[1] + C == 0:
                count += 1
        scored.append((count, l))
        
    scored.sort(key=lambda x: -x[0])
    
    # Pick top 27
    selected = [l for _, l in scored[:27]]
    lines = np.array(selected, dtype=np.int64)
    
    return points, lines

def _construct_pencil_k6_9():
    """
    Alternative: K_{6,9} structure.
    Points:
    6 points on x=0: (0, 0)..(0, 5)
    9 points on x=1: (1, 0)..(1, 8)
    ... wait, we need 54 points.
    
    Let's do 3 groups of 18?
    Or 2 groups of 27?
    
    Let's try:
    Points U: 6 points on x=0.
    Points V: 9 points on x=1.
    Points W: 39 points elsewhere?
    
    This is getting complicated. The grid is a solid baseline.
    Let's stick with the grid + local search.
    """
    pass

def _construct_random_lattice():
    """
    Generate a random set of 54 points in a small box.
    """
    points = []
    while len(points) < 54:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if (x, y) not in points:
            points.append((x, y))
    return np.array(points, dtype=np.int64)

def _construct_grid():
    """
    Standard grid construction.
    """
    # 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Lines: rows, cols, diagonals
    lines_set = set()
    # Rows
    for j in range(9):
        lines_set.add(_canon_line(0, 1, -j))
    # Cols
    for i in range(6):
        lines_set.add(_canon_line(1, 0, -i))
    # Diagonals
    for c in range(-8, 6):
        lines_set.add(_canon_line(1, -1, -c))
    for c in range(0, 14):
        lines_set.add(_canon_line(1, 1, -c))
        
    lines_list = list(lines_set)
    scored = []
    for l in lines_list:
        A, B, C = l
        count = 0
        for p in points:
            if A*p[0] + B*p[1] + C == 0:
                count += 1
        scored.append((count, l))
        
    scored.sort(key=lambda x: -x[0])
    selected = [l for _, l in scored[:27]]
    lines = np.array(selected, dtype=np.int64)
    
    return points, lines

def szemeredi_trotter_incidences():
    start_time = time.time()
    
    # 1. Initial Construction: Grid
    points, lines = _construct_grid()
    score = _incidence_count(points, lines)
    
    # 2. Local Search
    points, score = _local_search(points, lines, start_time)
    
    # 3. Try a different initial construction: Pencil-like
    # Let's try to construct a "near-pencil"
    # 53 points on a line, 1 point off.
    # Line y=0. Points (0,0)..(52,0). Point (0, 1).
    # Lines:
    # 1. y=0 (53 incidences)
    # 2. x=0 (2 incidences: (0,0), (0,1))
    # 3. x=1 (1 incidence)
    # ...
    # This is bad.
    
    # Let's try a "grid" with 3x18?
    # 3x18 = 54.
    # Rows: 18 lines, 3 pts each.
    # Cols: 3 lines, 18 pts each.
    # Diagonals?
    
    # Let's try 2x27?
    # 2x27 = 54.
    # Rows: 27 lines, 2 pts each.
    # Cols: 2 lines, 27 pts each.
    # This uses 29 lines. We have 27.
    # We can drop the 2 columns.
    # Then we have 27 lines (the rows).
    # Incidences: 27 * 2 = 54.
    # Grid 6x9 gave:
    # Rows: 9 * 6 = 54
    # Cols: 6 * 9 = 54
    # Diagonals: 
    #   Slope 1: 
    #     c=0: 6 pts
    #     c=1: 5 pts
    #     ...
    #   Sum of diagonals is roughly 54 * 2 / 2 ? No.
    #   Total points on slope 1 diagonals:
    #   Each point is on exactly one slope 1 diagonal.
    #   So sum of points on slope 1 diagonals = 54.
    #   We have 14 diagonals.
    #   If we pick the longest ones, we get more.
    #   Longest: 6, 5, 4, 3, 2, 1...
    #   We need 12 more lines (27 - 15 = 12).
    #   Top 6 slope 1: 6+5+4+3+2+1 = 21.
    #   Top 6 slope -1: 6+5+4+3+2+1 = 21.
    #   Total: 54 (rows) + 54 (cols) + 21 + 21 = 150.
    
    # So the grid score is around 150.
    # Local search should improve it.
    
    # Let's also try a random restart if time permits.
    if _time_left(start_time, 630, 30):
        for _ in range(5):
            if not _time_left(start_time, 630, 30):
                break
            # Perturb the grid
            p_copy = points.copy()
            for i in range(54):
                if random.random() < 0.5:
                    p_copy[i, 0] += random.randint(-2, 2)
                    p_copy[i, 1] += random.randint(-2, 2)
            
            # Check validity
            valid = True
            seen = set()
            for p in p_copy:
                if tuple(p) in seen:
                    valid = False
                    break
                seen.add(tuple(p))
            
            if valid:
                l_copy = _get_lines_from_points(p_copy)
                s_copy = _incidence_count(p_copy, l_copy)
                
                if s_copy > score:
                    points, lines, score = p_copy, l_copy, s_copy
                    points, score = _local_search(points, lines, start_time)
    
    return points, lines

# EVOLVE-BLOCK-END
