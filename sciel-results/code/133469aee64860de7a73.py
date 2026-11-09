# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) for A*x + B*y + C = 0."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _compute_top27(points, line_cache=None):
    """
    Compute the top 27 lines by incidence count for a set of points.
    Returns (score, lines_array) where score is the sum of incidences of the top 27 lines.
    Uses vectorized NumPy operations for speed.
    If line_cache is provided (a dict mapping line_tuple -> count), reuse it for lines that haven't changed.
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    pts = points
    
    # Step 1: Find all unique lines defined by pairs of points
    line_set = set()
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            # Line through (x1,y1) and (x2,y2)
            # (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - x1 * y2
            
            key = _normalize_line(A, B, C)
            if key is not None:
                line_set.add(key)
                
    # Step 2: Count incidences for each line using vectorized operations
    lines_list = list(line_set)
    lines_arr = np.array(lines_list, dtype=np.int64)
    m = len(lines_arr)
    
    if m == 0:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Vectorized: compute A*x + B*y + C for all (line, point) pairs
    A_col = lines_arr[:, 0:1]  # (m, 1)
    B_col = lines_arr[:, 1:2]  # (m, 1)
    C_col = lines_arr[:, 2:3]  # (m, 1)
    X_row = pts[:, 0:1].T      # (1, n)
    Y_row = pts[:, 1:2].T      # (1, n)
    
    # incidence_matrix[i,j] = A_i * x_j + B_i * y_j + C_i
    incidence_matrix = A_col * X_row + B_col * Y_row + C_col  # (m, n)
    
    # Count incidences: number of points where value == 0
    counts_vec = np.sum(incidence_matrix == 0, axis=1)  # (m,)
    
    # Step 3: Sort by count descending, then by line tuple for determinism
    order = np.lexsort((lines_arr[:, 2], lines_arr[:, 1], lines_arr[:, 0], -counts_vec))
    
    # Step 4: Select top 27
    k = min(27, m)
    selected_idx = order[:k]
    
    score = int(np.sum(counts_vec[selected_idx]))
    lines = lines_arr[selected_idx]
    
    return score, lines

def _generate_grid(a, b):
    """Generate an a x b grid of points starting at (1,1)."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _remove_points(points, indices):
    """Remove points at given indices."""
    mask = np.ones(len(points), dtype=bool)
    mask[list(indices)] = False
    return points[mask]

def _add_points(points, new_pts):
    """Add new points to the array."""
    new_arr = np.array(new_pts, dtype=np.int64)
    return np.vstack([points, new_arr])

def _check_valid_points(points):
    """Basic validity check for points."""
    if len(points) != 54:
        return False
    if points.shape != (54, 2):
        return False
    # Check for duplicates
    pt_set = set(map(tuple, points))
    if len(pt_set) != 54:
        return False
    return True

def _local_search(points, deadline, best_score, best_points, best_lines):
    """
    Perform local search by moving individual points to nearby integer coordinates.
    Uses best-improvement strategy per pass for better convergence.
    Returns updated best_score, best_points, best_lines.
    """
    n = len(points)
    # Expanded neighborhood: 8-neighbors + distance-2 moves
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1),
                 (-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2),
                 (-2,1),(2,1),(-1,2),(1,2),(-2,-1),(2,-1),(-1,-2),(1,-2)]
    
    current_points = points.copy()
    current_score, _ = _compute_top27(current_points)
    
    # Track seen configurations to avoid cycles
    seen_configs = set()
    seen_configs.add(tuple(map(tuple, current_points)))
    
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline:
        improved = False
        best_move_score = current_score
        best_move_idx = -1
        best_move_delta = None
        best_move_lines = None
        
        for idx in range(n):
            for dx, dy in neighbors:
                if time.time() > deadline:
                    return best_score, best_points, best_lines
                
                new_points = current_points.copy()
                new_points[idx] = current_points[idx] + np.array([dx, dy])
                
                if not _check_valid_points(new_points):
                    continue
                
                # Check for duplicate configuration
                config_key = tuple(map(tuple, new_points))
                if config_key in seen_configs:
                    continue
                    
                new_score, new_lines = _compute_top27(new_points)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_idx = idx
                    best_move_delta = (dx, dy)
                    best_move_lines = new_lines
        
        if best_move_idx >= 0:
            current_points[best_move_idx] = current_points[best_move_idx] + np.array(best_move_delta)
            current_score = best_move_score
            improved = True
            no_improve_count = 0
            seen_configs.add(tuple(map(tuple, current_points)))
            
            if best_move_score > best_score:
                best_score = best_move_score
                best_points = current_points.copy()
                best_lines = best_move_lines
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break

    return best_score, best_points, best_lines

def _line_based_search(points, deadline, best_score, best_points, best_lines):
    """
    Directly optimize the 27 line coefficients while keeping points fixed.
    For each line, try perturbing (A,B,C) to find lines with higher incidence.
    Returns updated best_score, best_points, best_lines.
    """
    if best_lines is None:
        return best_score, best_points, best_lines
    
    pts = best_points
    n = len(pts)
    current_lines = best_lines.copy()
    current_score = best_score
    
    # For each line, try small perturbations to its coefficients
    # A line (A,B,C) means A*x + B*y + C = 0
    # We can try scaling or shifting C, or small changes to A,B
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for li in range(27):
            A, B, C = current_lines[li]
            
            # Try various perturbations to this line
            candidates = []
            
            # Small changes to C (shift the line)
            for dC in [-2, -1, 1, 2, -3, 3]:
                new_C = C + dC
                if new_C != 0 or (A != 0 or B != 0):
                    candidates.append((A, B, new_C))
            
            # Small changes to A
            for dA in [-1, 1]:
                if A + dA != 0 or B != 0:
                    candidates.append((A + dA, B, C))
            
            # Small changes to B
            for dB in [-1, 1]:
                if B + dB != 0 or A != 0:
                    candidates.append((A, B + dB, C))
            
            # Try lines through pairs of points that are near the current line
            # Sample a few point pairs and check if they give better lines
            if time.time() < deadline:
                # Try all pairs (expensive but thorough for 54 points = 1431 pairs)
                # Only do this if we have time budget
                for i in range(n):
                    for j in range(i + 1, n):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        new_A = y2 - y1
                        new_B = -(x2 - x1)
                        new_C = x2 * y1 - x1 * y2
                        key = _normalize_line(new_A, new_B, new_C)
                        if key is not None:
                            candidates.append(key)
            
            # Evaluate each candidate
            best_cand_score = -1
            best_cand_line = None
            
            for cand in candidates:
                cA, cB, cC = cand
                # Count incidences for this single line
                count = np.sum(cA * pts[:, 0] + cB * pts[:, 1] + cC == 0)
                
                if count > best_cand_score:
                    best_cand_score = count
                    best_cand_line = cand
            
            if best_cand_line is not None:
                # Compute new total score if we replace line li with best_cand_line
                # Need to handle duplicates: if the new line is already in the set, skip
                old_line_norm = _normalize_line(int(A), int(B), int(C))
                new_line_norm = _normalize_line(best_cand_line[0], best_cand_line[1], best_cand_line[2])
                
                if old_line_norm != new_line_norm:
                    # Check if new_line_norm is already in current_lines
                    dup = False
                    for other in range(27):
                        if other == li:
                            continue
                        other_norm = _normalize_line(int(current_lines[other, 0]), 
                                                      int(current_lines[other, 1]), 
                                                      int(current_lines[other, 2]))
                        if other_norm == new_line_norm:
                            dup = True
                            break
                    
                    if not dup:
                        # Replace and compute new score
                        old_count = np.sum(A * pts[:, 0] + B * pts[:, 1] + C == 0)
                        new_total = current_score - old_count + best_cand_score
                        
                        if new_total > current_score:
                            current_lines[li] = best_cand_line
                            current_score = new_total
                            improved = True
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_points = pts.copy()
                                best_lines = current_lines.copy()
                                break  # Restart with new configuration
    
    return best_score, best_points, best_lines


def _simulated_annealing(points, deadline, best_score, best_points, best_lines):
    """
    Simulated annealing to escape local optima.
    Uses larger jumps and adaptive cooling.
    """
    n = len(points)
    current = points.copy()
    current_score, _ = _compute_top27(current)
    
    T = 20.0
    T_min = 1.0
    cooling = 0.985
    max_iter = 5000
    
    iteration = 0
    while time.time() < deadline and T > T_min and iteration < max_iter:
        iteration += 1
        # Pick random point and random move
        idx = np.random.randint(n)
        # Use larger jumps early, smaller later
        jump_size = max(1, int(4 * (T / 20.0)))
        dx = np.random.randint(-jump_size, jump_size + 1)
        dy = np.random.randint(-jump_size, jump_size + 1)
        if dx == 0 and dy == 0:
            T *= cooling
            continue
        
        new_points = current.copy()
        new_points[idx] = current[idx] + np.array([dx, dy])
        
        if not _check_valid_points(new_points):
            T *= cooling
            continue
        
        new_score, new_lines = _compute_top27(new_points)
        delta = new_score - current_score
        
        if delta > 0 or (T > 0 and np.random.random() < np.exp(delta / T)):
            current = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                best_lines = new_lines
        
        T *= cooling
    
    return best_score, best_points, best_lines

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Uses a combination of grid configurations, local search, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    np.random.seed(42)
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score, lines):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines

    # Phase 1: Systematic grid search with modifications
    # The current best (score 1.32) is likely the 6x9 grid.
    # Let's try various grids and perturbations.
    
    grid_configs = [
        # (a, b, remove_indices, add_points)
        (6, 9, [], []),
        (9, 6, [], []),
        (3, 18, [], []),
        (18, 3, [], []),
        (2, 27, [], []),
        (27, 2, [], []),
        (5, 11, [], []),  # 55 points, need to remove 1
        (11, 5, [], []),  # 55 points, need to remove 1
        (7, 8, [0], []),  # 56 points, remove 1
        (7, 8, [55], []),
        (7, 8, [0, 55], []), # 56 points, remove 2? No, 7*8=56. 56-2=54.
        (4, 14, [0], []), # 56 points, remove 1
        (4, 14, [0, 55], []), # 56 points, remove 2
        (5, 10, [], [(1,11),(2,11),(3,11),(4,11)]),  # 50 + 4
        (5, 10, [], [(6,1),(6,2),(6,3),(6,4)]),      # 50 + 4
        (10, 5, [], [(11,1),(11,2),(11,3),(11,4)]),  # 50 + 4
        (10, 5, [], [(1,6),(2,6),(3,6),(4,6)]),      # 50 + 4
        (4, 12, [], [(1,13),(2,13),(3,13),(4,13),(5,13),(6,13)]), # 48 + 6
        (4, 12, [], [(5,1),(5,2),(5,3),(5,4),(5,5),(5,6)]),       # 48 + 6
        (3, 17, [], [(1,18),(2,18),(3,18),(4,18),(5,18),(6,18),(7,18)]), # 51 + 3? 3*17=51. Need 3 more.
        (3, 17, [], [(4,1),(4,2),(4,3)]), # 51 + 3
    ]
    
    for a, b, rem, add in grid_configs:
        if time.time() > deadline - 120:
            break
        pts = _generate_grid(a, b)
        if rem:
            pts = _remove_points(pts, rem)
        if add:
            pts = _add_points(pts, add)
        if len(pts) != 54:
            continue
        if not _check_valid_points(pts):
            continue
        score, lines = _compute_top27(pts)
        update_best(pts, score, lines)
        
    # Phase 2: Local search from best grid
    if best_points is not None and time.time() < deadline - 180:
        best_score, best_points, best_lines = _local_search(
            best_points, deadline - 120, best_score, best_points, best_lines
        )
    
    # Phase 2b: Line-based search - directly optimize the 27 lines
    if best_points is not None and time.time() < deadline - 90:
        best_score, best_points, best_lines = _line_based_search(
            best_points, deadline - 60, best_score, best_points, best_lines
        )
    
    # Phase 3: Simulated Annealing to explore further
    if best_points is not None and time.time() < deadline - 30:
        best_score, best_points, best_lines = _simulated_annealing(
            best_points, deadline - 10, best_score, best_points, best_lines
        )
    
    # Phase 4: Final local search polish
    if best_points is not None and time.time() < deadline - 5:
        best_score, best_points, best_lines = _local_search(
            best_points, deadline - 2, best_score, best_points, best_lines
        )
        
    # Fallback if something went wrong
    if best_points is None:
        pts = _generate_grid(6, 9)
        score, lines = _compute_top27(pts)
        return pts, lines

    return best_points, best_lines

# EVOLVE-BLOCK-END
