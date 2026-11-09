import numpy as np
import time
import random
import math
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _count_incidences(points, lines):
    # Vectorized count
    # points: (N, 2), lines: (M, 3)
    x = points[:, 0].astype(np.int64)
    y = points[:, 1].astype(np.int64)
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    
    # We want sum over i,j of (A_j*x_i + B_j*y_i + C_j == 0)
    # Using broadcasting: (M, 1) * (1, N) -> (M, N)
    # But M=27, N=54, so 27*54 = 1458 elements. Very fast.
    vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
    return np.sum(vals == 0)

def _generate_lines_for_points(points):
    """
    Generate all unique lines passing through at least 2 points.
    Returns a list of (count, line_tuple) sorted by count desc.
    """
    n = len(points)
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key not in line_counts:
                line_counts[key] = 0
            
    # Now count incidences for each unique line
    # To be efficient, we can just iterate points for each line?
    # Number of unique lines is at most n(n-1)/2 = 1431.
    # 1431 * 54 = ~77k ops.
    
    for key in line_counts:
        A, B, C = key
        cnt = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        line_counts[key] = cnt
        
    # Sort by count desc, then by key for determinism
    sorted_lines = sorted(line_counts.items(), key=lambda item: (-item[1], item[0]))
    return sorted_lines

def _best_lines_from_sorted(sorted_lines, k=27):
    """
    Takes sorted list of (count, key), returns top k keys as numpy array.
    """
    top_keys = [k for _, k in sorted_lines[:k]]
    # If we have fewer than k lines (should not happen for 54 distinct points), pad?
    # But we must return valid lines.
    # With 54 points, we have >= 27 lines.
    return np.array(top_keys, dtype=np.int64)

def _local_search(points, lines, time_limit):
    """
    Perform local search by swapping lines and moving points.
    Returns best points, lines, score.
    """
    start = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = _count_incidences(current_points, current_lines)
    
    # Precompute sorted lines for current points
    sorted_lines = _generate_lines_for_points(current_points)
    
    # We maintain a set of "candidate" lines: the top 50 or so, plus the current 27.
    # Actually, just recompute sorted_lines when points change.
    
    best_score = current_score
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    
    # Local search loop
    # 1. Try swapping a current line with a non-current line from the top candidates.
    # 2. Try moving a point.
    
    while time.time() - start < time_limit:
        improved = False
        
        # --- Phase 1: Line Swaps ---
        # Get top 50 lines for current points
        top_candidates = _generate_lines_for_points(current_points)
        candidate_keys = [k for _, k in top_candidates[:50]]
        
        # Current lines
        current_line_keys = [tuple(l) for l in current_lines]
        
        # Try replacing each current line with a better candidate
        for i in range(27):
            # Find best candidate not in current set
            best_cand = None
            best_cand_cnt = -1
            for k in candidate_keys:
                if k not in current_line_keys:
                    cnt = dict(top_candidates)[k]
                    if cnt > best_cand_cnt:
                        best_cand_cnt = cnt
                        best_cand = k
            
            if best_cand is None:
                continue
                
            # What is the count of the line being removed?
            old_line = current_lines[i]
            old_cnt = 0
            A, B, C = old_line
            for p in current_points:
                x, y = p
                if A * x + B * y + C == 0:
                    old_cnt += 1
            
            if best_cand_cnt > old_cnt:
                # Swap
                new_lines = current_lines.copy()
                new_lines[i] = best_cand
                new_score = _count_incidences(current_points, new_lines)
                
                if new_score > current_score:
                    current_lines = new_lines
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                        best_lines = current_lines.copy()
                    # Break to restart line swap phase with new lines?
                    # Or continue? Restarting is safer to update candidate list.
                    break
        
        if improved:
            continue # Restart loop to re-evaluate candidates
        
        # --- Phase 2: Point Moves ---
        # Try moving each point by a small vector
        # To be fast, we only evaluate if the move is promising?
        # Or just try a few random moves.
        
        # Deterministic hill climbing on points is hard because line set changes.
        # Let's try random point perturbations but limit the number.
        
        n_moves = 0
        max_moves = 20 # Limit per iteration to allow line swaps to be frequent
        
        while n_moves < max_moves and time.time() - start < time_limit:
            idx = random.randint(0, 53)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                n_moves += 1
                continue
            
            new_points = current_points.copy()
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
            
            # Check duplicates
            valid = True
            pt_set = set()
            for p in new_points:
                key = (int(p[0]), int(p[1]))
                if key in pt_set:
                    valid = False
                    break
                pt_set.add(key)
            
            if not valid:
                n_moves += 1
                continue
            
            # Find best lines for new points
            new_sorted = _generate_lines_for_points(new_points)
            new_lines = _best_lines_from_sorted(new_sorted, 27)
            
            # Check validity (uniqueness of lines is guaranteed by set in _generate)
            # Check A!=0 or B!=0 (guaranteed by construction from distinct points)
            
            new_score = _count_incidences(new_points, new_lines)
            
            if new_score > current_score:
                current_points = new_points
                current_lines = new_lines
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            
            n_moves += 1
        
        if improved:
            continue
            
        # If no improvement in either phase, we might be stuck in a local optimum.
        # We could break here and let the outer loop try a different initialization.
        # Or do a random restart?
        # For now, break to let outer loop handle it.
        break
        
    return best_points, best_lines, best_score

def _generate_grid_points(w, h, offset_x=0, offset_y=0):
    points = []
    for i in range(w):
        for j in range(h):
            points.append((i + offset_x, j + offset_y))
    return np.array(points, dtype=np.int64)

def _generate_triangular_points(n):
    # Pack points in a roughly circular/triangular shape
    points = []
    # Simple spiral or grid crop
    # Let's use a grid and crop to n points
    side = int(math.ceil(math.sqrt(n)))
    for i in range(side):
        for j in range(side):
            if len(points) < n:
                points.append((i, j))
    return np.array(points[:n], dtype=np.int64)

def _generate_product_points(n1, n2):
    # Points from product of two sets
    # e.g., x in {0..n1-1}, y in {0..n2-1}
    # This is just a grid.
    # Let's try a "dense" grid like 6x9
    return _generate_grid_points(n1, n2)

def _generate_random_points(n, limit=20):
    points = []
    seen = set()
    while len(points) < n:
        x = random.randint(-limit, limit)
        y = random.randint(-limit, limit)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append((x, y))
    return np.array(points, dtype=np.int64)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points_global = None
    best_lines_global = None
    best_score_global = 0
    
    # List of initializations to try
    inits = []
    
    # 1. Standard grids
    inits.append(_generate_grid_points(6, 9))
    inits.append(_generate_grid_points(9, 6))
    inits.append(_generate_grid_points(7, 8)) # 56 points, need to trim or adjust
    # 7x8 = 56. We need 54. Remove 2 corners.
    p78 = _generate_grid_points(7, 8)
    # Remove (0,0) and (6,7)
    mask = ~((p78[:,0]==0)&(p78[:,1]==0)) & ~((p78[:,0]==6)&(p78[:,1]==7))
    inits.append(p78[mask])
    
    inits.append(_generate_grid_points(5, 11)) # 55, remove 1
    p511 = _generate_grid_points(5, 11)
    mask = ~((p511[:,0]==0)&(p511[:,1]==0))
    inits.append(p511[mask])
    
    inits.append(_generate_grid_points(4, 14)) # 56, remove 2
    p414 = _generate_grid_points(4, 14)
    mask = ~((p414[:,0]==0)&(p414[:,1]==0)) & ~((p414[:,0]==3)&(p414[:,1]==13))
    inits.append(p414[mask])
    
    inits.append(_generate_grid_points(3, 18)) # 54 exactly
    inits.append(_generate_grid_points(2, 27)) # 54 exactly
    
    # 2. Triangular/Circular
    inits.append(_generate_triangular_points(54))
    
    # 3. Random
    for _ in range(3):
        inits.append(_generate_random_points(54, limit=15))
        
    # 4. "Diagonal" heavy grid
    # Start with 6x9 and shift some rows
    p_shift = _generate_grid_points(6, 9).copy()
    for i in range(1, 9, 2):
        p_shift[i*6:(i+1)*6, 0] += 1 # Shift odd rows
    inits.append(p_shift)
    
    # Sort inits by initial score (best first) to prioritize good starts
    scored_inits = []
    for p in inits:
        if p.shape != (54, 2):
            continue
        # Check validity of points
        if not _is_valid(p, np.zeros((27,3), dtype=np.int64)): # Just check points part of is_valid? 
            # _is_valid checks lines too. Let's just check points manually or assume valid for now.
            # Actually _is_valid returns False if lines are invalid.
            # Let's just generate lines and check.
            pass
        
        sl = _generate_lines_for_points(p)
        lines = _best_lines_from_sorted(sl, 27)
        if not _is_valid(p, lines):
            continue
        score = _count_incidences(p, lines)
        scored_inits.append((score, p, lines))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    # Allocate time to each init
    # Give more time to better inits
    # Total time ~ 580s
    # If we have 10 inits, maybe 50s each?
    # Or adaptive: run until improvement stops, then move to next.
    
    for score, p, l in scored_inits:
        if time.time() > deadline - 10: # Reserve 10s for final check
            break
            
        if score > best_score_global:
            best_score_global = score
            best_points_global = p.copy()
            best_lines_global = l.copy()
            
        # Run local search
        time_alloc = 30.0 # 30 seconds per init
        # If this is the best so far, give it more time?
        if score == scored_inits[0][0]:
             time_alloc = 60.0
             
        p_res, l_res, s_res = _local_search(p, l, time_alloc)
        
        if s_res > best_score_global:
            best_score_global = s_res
            best_points_global = p_res.copy()
            best_lines_global = l_res.copy()
            
    # Final fallback if no valid result
    if best_points_global is None:
        p = _generate_grid_points(6, 9)
        sl = _generate_lines_for_points(p)
        l = _best_lines_from_sorted(sl, 27)
        best_points_global = p
        best_lines_global = l
        best_score_global = _count_incidences(p, l)
        
    return best_points_global, best_lines_global

# Quick self-test if run directly
if __name__ == "__main__":
    p, l = szemeredi_trotter_incidences()
    print("Score:", _count_incidences(p, l))
    print("Valid:", _is_valid(p, l))
