# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import itertools

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    total = px * A + py * B + C
    return np.sum(total == 0)

def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1

def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _generate_candidates_for_grid(x_min, x_max, y_min, y_max, max_coeff=4):
    candidates = []
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            
            cx_min = 0
            cx_max = 0
            if A > 0:
                cx_min = -A * x_max
                cx_max = -A * x_min
            elif A < 0:
                cx_min = -A * x_min
                cx_max = -A * x_max
            
            cy_min = 0
            cy_max = 0
            if B > 0:
                cy_min = -B * y_max
                cy_max = -B * y_min
            elif B < 0:
                cy_min = -B * y_min
                cy_max = -B * y_max
            
            C_min = cx_min + cy_min
            C_max = cx_max + cy_max
            
            for C in range(C_min, C_max + 1):
                candidates.append((A, B, C))
    return candidates

def _select_best_lines(points, candidates, k=27):
    if not candidates:
        return [], 0
        
    cand_array = np.array(candidates, dtype=np.int64)
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = cand_array[:, 0:1].T
    B = cand_array[:, 1:2].T
    C = cand_array[:, 2:3].T
    
    total = px * A + py * B + C
    incidences = np.sum(total == 0, axis=0)
    
    sorted_indices = np.argsort(incidences)[::-1]
    
    selected_lines = []
    seen_normalized = set()
    
    for idx in sorted_indices:
        if len(selected_lines) >= k:
            break
        A, B, C = cand_array[idx]
        norm = _normalize_line(A, B, C)
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            selected_lines.append(norm)
            
    if len(selected_lines) < k:
        # Fallback: pad with arbitrary valid lines if needed (should not happen for dense grids)
        # But we must ensure validity.
        # We can just return what we have, but the task requires 27 lines.
        # If we don't have 27, we might fail.
        # However, with 54 points and small coefficients, we should have enough.
        pass
        
    # Calculate total incidence for these lines
    if not selected_lines:
        return [], 0
        
    sel_array = np.array(selected_lines, dtype=np.int64)
    total_sel = np.sum(_incidence_count(points, sel_array))
    
    return selected_lines, total_sel

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Tries 6x9 grid and 7x8 grid (with 2 points removed).
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Strategy 1: 6x9 Grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_69 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    candidates_69 = _generate_candidates_for_grid(1, 6, 1, 9, max_coeff=4)
    lines_69, score_69 = _select_best_lines(points_69, candidates_69, 27)
    
    if score_69 > best_score:
        best_score = score_69
        best_points = points_69
        best_lines = lines_69
        
    # Strategy 2: 7x8 Grid (remove 2 points)
    # Full grid
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_full_78 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    candidates_78 = _generate_candidates_for_grid(1, 7, 1, 8, max_coeff=4)
    
    # Identify points to remove.
    # Heuristic: Remove corners or points with low "centrality".
    # Let's try removing (1,1) and (7,8), (1,8) and (7,1), etc.
    # We will test a few specific pairs.
    
    points_list = points_full_78.tolist()
    
    # Indices of corners
    # (1,1) is idx 0
    # (7,1) is idx 6
    # (1,8) is idx 49
    # (7,8) is idx 55
    
    removal_pairs = [
        (0, 55), (0, 49), (6, 55), (6, 49),
        (0, 6), (49, 55), # Adjacent corners
        (0, 27), (55, 27) # Corner and center-ish
    ]
    
    for i, j in removal_pairs:
        if i == j: continue
        # Create subset
        mask = np.ones(56, dtype=bool)
        mask[i] = False
        mask[j] = False
        points_subset = points_full_78[mask]
        
        if len(points_subset) != 54: continue
        
        lines_subset, score_subset = _select_best_lines(points_subset, candidates_78, 27)
        
        if score_subset > best_score:
            best_score = score_subset
            best_points = points_subset
            best_lines = lines_subset
            
        if time.time() > deadline - 10:
            break
            
    # Strategy 3: 5x11 Grid (remove 1 point)
    xs = np.arange(1, 6)
    ys = np.arange(1, 12)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_full_511 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    candidates_511 = _generate_candidates_for_grid(1, 5, 1, 11, max_coeff=4)
    
    # Remove 1 point. Try corners.
    removal_indices_511 = [0, 4, 51, 55] # (1,1), (5,1), (1,11), (5,11)
    
    for i in removal_indices_511:
        mask = np.ones(55, dtype=bool)
        mask[i] = False
        points_subset = points_full_511[mask]
        
        if len(points_subset) != 54: continue
        
        lines_subset, score_subset = _select_best_lines(points_subset, candidates_511, 27)
        
        if score_subset > best_score:
            best_score = score_subset
            best_points = points_subset
            best_lines = lines_subset
            
        if time.time() > deadline - 10:
            break
            
    if best_points is None or best_lines is None:
        # Fallback to 6x9
        return points_69, np.array(lines_69, dtype=np.int64)
        
    return best_points, np.array(best_lines, dtype=np.int64)

# EVOLVE-BLOCK-END
