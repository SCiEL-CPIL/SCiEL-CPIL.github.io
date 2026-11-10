# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
import random

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    # Vectorized check: A*x + B*y + C == 0
    # points: (N, 2), lines: (M, 3)
    # We want (N, M) boolean matrix
    # (N, 1) * (1, M) + (N, 1) * (1, M) + (1, M)
    px = points[:, 0:1]  # (N, 1)
    py = points[:, 1:2]  # (N, 1)
    A = lines[:, 0][np.newaxis, :]  # (1, M)
    B = lines[:, 1][np.newaxis, :]  # (1, M)
    C = lines[:, 2][np.newaxis, :]  # (1, M)
    total = px * A + py * B + C  # (N, M)
    return int(np.sum(total == 0))

def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(line):
    return _normalize_line(*line)

def _is_valid_configuration(points, lines):
    """Check validity constraints."""
    # 1. Shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # 2. Integer check (entries should be close to integers, then rounded)
    # The problem says entries are rounded to int64. 
    # If we generate integer points/lines, they are valid.
    # We ensure we only work with integers in our search space.
    
    # 3. No repeated points
    # Check unique points
    # Sort points to check for duplicates efficiently
    sorted_pts = np.lexsort((points[:, 1], points[:, 0]))
    if np.any(np.diff(sorted_pts[:, 0]) == 0) and np.any(np.diff(sorted_pts[:, 1]) == 0):
        # Need to check pairs
        for i in range(len(points) - 1):
            if sorted_pts[i, 0] == sorted_pts[i+1, 0] and sorted_pts[i, 1] == sorted_pts[i+1, 1]:
                return False

    # 4. No invalid lines (A=B=0)
    for line in lines:
        if line[0] == 0 and line[1] == 0:
            return False
            
    # 5. No duplicate lines
    seen_lines = set()
    for line in lines:
        key = _line_key(line)
        if key in seen_lines:
            return False
        seen_lines.add(key)
        
    return True

def _generate_initial_clique_star():
    """
    Generate initial configuration:
    27 points on a vertical line (x=0, y=0..26)
    27 points on a 3x9 grid (x=1..3, y=0..8)
    Lines: 
    1 line for the vertical clique (x=0)
    26 lines for the grid (mix of horizontal, vertical, diagonal)
    """
    points = np.zeros((54, 2), dtype=np.int64)
    
    # Clique part
    for i in range(27):
        points[i] = [0, i]
        
    # Grid part
    grid_x = np.array([1, 2, 3])
    grid_y = np.array(range(9))
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")
    grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    points[27:] = grid_points
    
    # Lines
    lines = np.zeros((27, 3), dtype=np.int64)
    
    # Line 0: x=0 (covers all 27 clique points)
    lines[0] = [1, 0, 0]
    
    # Lines 1-26: Cover grid points
    # Grid is 3x9. 
    # 3 vertical lines: x=1, x=2, x=3 (9 points each)
    lines[1] = [1, 0, -1]
    lines[2] = [1, 0, -2]
    lines[3] = [1, 0, -3]
    
    # 9 horizontal lines: y=0..8 (3 points each)
    for i in range(9):
        lines[4 + i] = [0, 1, -i]
        
    # 14 remaining lines: Diagonals
    # Slope 1: y = x + b
    # Points (1,0)..(3,8)
    # b = y - x
    # x in 1..3, y in 0..8
    # b in -3..7
    # Let's pick lines that cover many points.
    # y = x - 1 (1,0), (2,1), (3,2) -> 3 pts
    # y = x - 2 (2,0), (3,1) -> 2 pts
    # y = x - 3 (3,0) -> 1 pt
    # y = x (1,1)..(3,3) -> 3 pts
    # y = x + 1 (1,2)..(3,4) -> 3 pts
    # y = x + 2 (1,3)..(3,5) -> 3 pts
    # y = x + 3 (1,4)..(3,6) -> 3 pts
    # y = x + 4 (1,5)..(3,7) -> 3 pts
    # y = x + 5 (1,6)..(3,8) -> 3 pts
    # y = x + 6 (1,7), (2,8) -> 2 pts
    # y = x + 7 (1,8) -> 1 pt
    
    # We have 14 slots. Let's fill with best diagonals.
    diag_candidates = []
    for b in range(-3, 8):
        # Count incidences in grid
        # A=1, B=-1, C=b
        inc = 0
        for gx_i, gy_i in zip(gx.ravel(), gy.ravel()):
            if 1 * gx_i - 1 * gy_i + b == 0:
                inc += 1
        diag_candidates.append((inc, b))
        
    # Sort by incidence descending
    diag_candidates.sort(reverse=True)
    
    idx = 13
    for inc, b in diag_candidates[:14]:
        lines[idx] = [1, -1, b]
        idx += 1
        
    return points, lines

def _local_search(points, lines, time_limit):
    """
    Perform local search by swapping points and lines.
    """
    best_score = _incidence_count(points, lines)
    current_points = points.copy()
    current_lines = lines.copy()
    
    start_time = time.time()
    
    # Pre-compute incidence matrix for speed?
    # It's 54x27 = 1458 entries. Very small.
    # We can recompute quickly.
    
    # Mutation strategies:
    # 1. Move a point to a random nearby integer location
    # 2. Swap two points
    # 3. Modify a line's coefficients slightly
    # 4. Swap two lines
    
    # To make this effective, we need to be smart about candidates.
    # Let's try a simple hill climbing with random restarts if stuck.
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try random mutations
        # 1. Point move
        if random.random() < 0.5:
            p_idx = random.randint(0, 53)
            old_p = current_points[p_idx].copy()
            # Move to a random location in a box around current
            # Or just a random location in a large box
            new_x = old_p[0] + random.randint(-5, 5)
            new_y = old_p[1] + random.randint(-5, 5)
            
            # Check if new point is valid (integer, not duplicate)
            # We'll just try it and check validity
            current_points[p_idx] = [new_x, new_y]
            
            if _is_valid_configuration(current_points, current_lines):
                new_score = _incidence_count(current_points, current_lines)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    no_improve_count = 0
                else:
                    current_points[p_idx] = old_p
                    no_improve_count += 1
            else:
                current_points[p_idx] = old_p
                no_improve_count += 1
        else:
            # 2. Line modification
            l_idx = random.randint(0, 26)
            old_l = current_lines[l_idx].copy()
            
            # Perturb A, B, C
            # Keep A, B small
            new_A = old_l[0] + random.randint(-2, 2)
            new_B = old_l[1] + random.randint(-2, 2)
            new_C = old_l[2] + random.randint(-10, 10)
            
            # Ensure A, B not both 0
            if new_A == 0 and new_B == 0:
                new_A = 1
            
            current_lines[l_idx] = [new_A, new_B, new_C]
            
            if _is_valid_configuration(current_points, current_lines):
                new_score = _incidence_count(current_points, current_lines)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    no_improve_count = 0
                else:
                    current_lines[l_idx] = old_l
                    no_improve_count += 1
            else:
                current_lines[l_idx] = old_l
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Restart from best known
            # Maybe try a different initial configuration?
            # For now, just break and let outer loop handle restart
            break
            
    return current_points, current_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try multiple initializations
    # 1. Clique-Star
    # 2. Pure Grid 6x9
    # 3. Pure Grid 9x6
    
    initializations = []
    
    # Init 1: Clique-Star
    p1, l1 = _generate_initial_clique_star()
    initializations.append((p1, l1))
    
    # Init 2: 6x9 Grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    p2 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate lines for grid
    l2 = np.zeros((27, 3), dtype=np.int64)
    # 6 vertical
    for i in range(6):
        l2[i] = [1, 0, -(i+1)]
    # 9 horizontal
    for i in range(9):
        l2[6+i] = [0, 1, -(i+1)]
    # 12 diagonals (slope 1)
    for i in range(12):
        b = i - 5 # range -5..6
        l2[15+i] = [1, -1, b]
        
    initializations.append((p2, l2))
    
    # Init 3: 9x6 Grid
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    p3 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    l3 = np.zeros((27, 3), dtype=np.int64)
    # 9 vertical
    for i in range(9):
        l3[i] = [1, 0, -(i+1)]
    # 6 horizontal
    for i in range(6):
        l3[9+i] = [0, 1, -(i+1)]
    # 12 diagonals
    for i in range(12):
        b = i - 8 # range -8..3
        l3[15+i] = [1, -1, b]
        
    initializations.append((p3, l3))
    
    # Run local search on each initialization
    # Allocate time based on number of inits
    time_per_init = (deadline - start_time - 10) / len(initializations)
    
    for p_init, l_init in initializations:
        if time.time() > deadline - 10:
            break
            
        p_curr, l_curr, score = _local_search(p_init, l_init, time_per_init)
        
        if score > best_score:
            best_score = score
            best_points = p_curr
            best_lines = l_curr
            
    # Fallback if something went wrong
    if best_points is None:
        best_points, best_lines = _generate_initial_clique_star()
        
    # Final validation and cleanup
    # Ensure integer types
    best_points = best_points.astype(np.int64)
    best_lines = best_lines.astype(np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
