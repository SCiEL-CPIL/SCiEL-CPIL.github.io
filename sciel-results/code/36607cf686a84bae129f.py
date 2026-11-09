import numpy as np
import time
import math
from math import gcd
import random

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

def _compute_top27(points):
    """
    Compute the top 27 lines by incidence count for a set of points.
    Uses vectorized incidence counting for speed.
    Returns (score, lines_array) where score is the sum of incidences of the top 27 lines.
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    pts = np.asarray(points, dtype=np.int64)
    
    # Step 1: Find all unique lines defined by pairs of points
    line_set = set()
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - x1 * y2
            
            key = _normalize_line(A, B, C)
            if key is not None:
                line_set.add(key)
                
    if not line_set:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Step 2: Vectorized incidence counting
    lines_arr = np.array(list(line_set), dtype=np.int64)  # (m, 3)
    # Compute A*x + B*y + C for all (line, point) pairs
    # lines_arr[:, 0] * pts[:, 0] + lines_arr[:, 1] * pts[:, 1] + lines_arr[:, 2]
    # Shape: (m, n)
    incidences = (lines_arr[:, 0:1] * pts[0:1, 0:1] + 
                  lines_arr[:, 1:2] * pts[0:1, 1:2] + 
                  lines_arr[:, 2:3])
    counts_arr = np.sum(incidences == 0, axis=1)  # (m,)
    
    # Step 3: Get top 27
    m = len(lines_arr)
    if m >= 27:
        # Get indices of top 27 by count, breaking ties by line tuple
        # Sort by (-count, A, B, C)
        sort_keys = np.lexsort((lines_arr[:, 2], lines_arr[:, 1], lines_arr[:, 0], -counts_arr))
        top_indices = sort_keys[:27]
        selected_counts = counts_arr[top_indices]
        selected_lines = lines_arr[top_indices]
    else:
        selected_counts = counts_arr
        selected_lines = lines_arr
        # Pad if needed
        pad_n = 27 - m
        if pad_n > 0:
            selected_counts = np.concatenate([selected_counts, np.zeros(pad_n, dtype=np.int64)])
            selected_lines = np.concatenate([selected_lines, np.zeros((pad_n, 3), dtype=np.int64)])
    
    score = int(np.sum(selected_counts))
    lines = selected_lines
    
    return score, lines

def _generate_grid(a, b):
    """Generate an a x b grid of points starting at (0,0)."""
    xs = np.arange(0, a)
    ys = np.arange(0, b)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

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
    Uses best-improvement strategy within each pass for better convergence.
    Returns updated best_score, best_points, best_lines.
    """
    n = len(points)
    # Neighborhood: 8-neighbors
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    
    current_points = points.copy()
    current_score, _ = _compute_top27(current_points)
    
    max_passes = 50
    for _pass in range(max_passes):
        if time.time() > deadline - 5:
            break
        
        improved = False
        best_move = None
        best_move_score = current_score
        
        for idx in range(n):
            if time.time() > deadline - 5:
                break
            for dx, dy in neighbors:
                new_points = current_points.copy()
                new_points[idx] = current_points[idx] + np.array([dx, dy])
                
                if not _check_valid_points(new_points):
                    continue
                    
                new_score, _ = _compute_top27(new_points)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (idx, dx, dy, new_points)
        
        if best_move is not None:
            idx, dx, dy, new_points = best_move
            current_points = new_points
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                best_lines, _ = _compute_top27(best_points)
            improved = True
        
        if not improved:
            break

    return best_score, best_points, best_lines

def _simulated_annealing(points, deadline, best_score, best_points, best_lines):
    """
    Simulated annealing with adaptive step size and restarts to escape local optima.
    """
    n = len(points)
    current = points.copy()
    current_score, _ = _compute_top27(current)
    
    T = 15.0
    T_min = 0.05
    cooling = 0.98
    
    # Track best seen during SA
    sa_best_score = current_score
    sa_best_points = current.copy()
    sa_best_lines, _ = _compute_top27(sa_best_points)
    
    steps_since_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10 and T > T_min:
        # Pick random point and random move with adaptive step size
        idx = np.random.randint(n)
        # Step size decreases with temperature
        max_step = max(1, int(5 * T / 15.0))
        dx = np.random.randint(-max_step, max_step + 1)
        dy = np.random.randint(-max_step, max_step + 1)
        if dx == 0 and dy == 0:
            T *= cooling
            continue
        
        new_points = current.copy()
        new_points[idx] = current[idx] + np.array([dx, dy])
        
        if not _check_valid_points(new_points):
            T *= cooling
            continue
        
        new_score, _ = _compute_top27(new_points)
        delta = new_score - current_score
        
        accept = False
        if delta > 0:
            accept = True
        elif T > 0:
            # Avoid overflow in exp
            if delta / T > -700:
                accept = np.random.random() < np.exp(delta / T)
        
        if accept:
            current = new_points
            current_score = new_score
            if new_score > sa_best_score:
                sa_best_score = new_score
                sa_best_points = current.copy()
                sa_best_lines, _ = _compute_top27(sa_best_points)
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            steps_since_improve += 1
        
        T *= cooling
        
        # Restart if stuck too long
        if steps_since_improve > max_no_improve:
            current = sa_best_points.copy()
            current_score = sa_best_score
            T = 10.0
            steps_since_improve = 0
    
    # Update global best
    if sa_best_score > best_score:
        best_score = sa_best_score
        best_points = sa_best_points.copy()
        best_lines = sa_best_lines
    
    return best_score, best_points, best_lines

def _line_based_optimize(points, deadline, best_score, best_points, best_lines):
    """
    Try to improve by replacing lines: for each current line, try to find a better
    line that captures more points. This explores the line space directly.
    """
    n = len(points)
    pts = np.asarray(points, dtype=np.int64)
    
    # Get current top 27 lines
    _, current_lines = _compute_top27(pts)
    
    # For each line slot, try alternative lines defined by pairs of points
    # that have high incidence
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Recompute current lines
        _, cur_lines = _compute_top27(pts)
        
        for line_idx in range(27):
            if time.time() > deadline - 5:
                break
            
            # Try to find a line that captures more points than cur_lines[line_idx]
            # Sample some pairs of points and check if the line through them
            # has higher incidence than the current line at this position
            cur_count = 0
            A, B, C = cur_lines[line_idx]
            for p in pts:
                if A * p[0] + B * p[1] + C == 0:
                    cur_count += 1
            
            # Try random pairs
            for _attempt in range(20):
                if time.time() > deadline - 5:
                    break
                i, j = np.random.choice(n, size=2, replace=False)
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A_new = y2 - y1
                B_new = -(x2 - x1)
                C_new = x2 * y1 - x1 * y2
                key = _normalize_line(A_new, B_new, C_new)
                if key is None:
                    continue
                # Check if this line is already in the top 27
                already_in = False
                for k in range(27):
                    if tuple(cur_lines[k]) == key:
                        already_in = True
                        break
                if already_in:
                    continue
                # Count incidences
                new_count = 0
                for p in pts:
                    if key[0] * p[0] + key[1] * p[1] + key[2] == 0:
                        new_count += 1
                if new_count > cur_count:
                    # This line is better; update points to encourage this line
                    # Actually, we can't change lines directly - lines are determined by points
                    # So this approach doesn't directly help unless we move points
                    # Skip for now - line-based optimization would need to move points
                    pass
        
    return best_score, best_points, best_lines

def _multi_start_local_search(points, deadline, best_score, best_points, best_lines):
    """
    Perform local search from multiple perturbed starting points.
    """
    n = len(points)
    
    # Try several random perturbations
    num_starts = 5
    for start_idx in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        # Create a perturbed version
        perturbed = points.copy()
        for _ in range(3):  # Move 3 random points
            idx = np.random.randint(n)
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            perturbed[idx] = perturbed[idx] + np.array([dx, dy])
        
        if not _check_valid_points(perturbed):
            continue
        
        # Local search from this start
        best_score, best_points, best_lines = _local_search(
            perturbed, deadline - 20, best_score, best_points, best_lines
        )
    
    return best_score, best_points, best_lines

def _final_polish(points, deadline, best_score, best_points, best_lines):
    """
    Final high-precision polish: try all single-point moves with larger radius,
    then 2-point moves.
    """
    n = len(points)
    pts = points.copy()
    current_score, _ = _compute_top27(pts)
    
    # Try moves with radius 2-3
    for radius in [2, 3]:
        if time.time() > deadline - 5:
            break
        improved = True
        while improved and time.time() > deadline - 5:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 5:
                    break
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = pts.copy()
                        new_pts[idx] = pts[idx] + np.array([dx, dy])
                        if not _check_valid_points(new_pts):
                            continue
                        new_score, _ = _compute_top27(new_pts)
                        if new_score > current_score:
                            pts = new_pts
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_points = pts.copy()
                                best_lines, _ = _compute_top27(best_points)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Try 2-point moves (move two points simultaneously)
    if time.time() > deadline - 5:
        improved = True
        while improved and time.time() > deadline - 5:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    for dx1 in [-1, 0, 1]:
                        for dy1 in [-1, 0, 1]:
                            if dx1 == 0 and dy1 == 0:
                                continue
                            for dx2 in [-1, 0, 1]:
                                for dy2 in [-1, 0, 1]:
                                    if dx2 == 0 and dy2 == 0:
                                        continue
                                    new_pts = pts.copy()
                                    new_pts[i] = pts[i] + np.array([dx1, dy1])
                                    new_pts[j] = pts[j] + np.array([dx2, dy2])
                                    if not _check_valid_points(new_pts):
                                        continue
                                    new_score, _ = _compute_top27(new_pts)
                                    if new_score > current_score:
                                        pts = new_pts
                                        current_score = new_score
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_points = pts.copy()
                                            best_lines, _ = _compute_top27(best_points)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_score, best_points, best_lines

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Uses a combination of grid configurations, local search, simulated annealing,
    multi-start search, and final polish.
    """
    start_time = time.time()
    deadline = start_time + 585  # Safety margin for 630s limit
    
    np.random.seed(42)
    random.seed(42)
    
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
    grid_configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
    ]
    
    for a, b in grid_configs:
        if time.time() > deadline - 180:
            break
        pts = _generate_grid(a, b)
        if len(pts) != 54:
            continue
        if not _check_valid_points(pts):
            continue
        score, lines = _compute_top27(pts)
        update_best(pts, score, lines)
        
    # Phase 2: Local search from best grid
    if best_points is not None and time.time() < deadline - 120:
        best_score, best_points, best_lines = _local_search(
            best_points, deadline - 60, best_score, best_points, best_lines
        )
        
    # Phase 3: Multi-start local search
    if best_points is not None and time.time() < deadline - 60:
        best_score, best_points, best_lines = _multi_start_local_search(
            best_points, deadline - 30, best_score, best_points, best_lines
        )
        
    # Phase 4: Simulated Annealing to explore further
    if best_points is not None and time.time() < deadline - 30:
        best_score, best_points, best_lines = _simulated_annealing(
            best_points, deadline - 15, best_score, best_points, best_lines
        )
        
    # Phase 5: Final polish with larger moves and 2-point moves
    if best_points is not None and time.time() < deadline - 5:
        best_score, best_points, best_lines = _final_polish(
            best_points, deadline, best_score, best_points, best_lines
        )
        
    # Fallback if something went wrong
    if best_points is None:
        pts = _generate_grid(6, 9)
        score, lines = _compute_top27(pts)
        return pts, lines

    return best_points, best_lines
