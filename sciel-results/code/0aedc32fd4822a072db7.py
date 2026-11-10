import numpy as np
import time
import random
from math import gcd

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences exactly."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    # (n, 1) * (1, m) + (n, 1) * (1, m) + (1, m) -> (n, m)
    vals = px * A + py * B + C
    return np.sum(vals == 0)

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    # If A, B, C are all 0, it's not a valid line.
    if A == 0 and B == 0 and C == 0:
        return None
    
    # Divide by GCD
    g = gcd(abs(A), abs(B))
    if g > 1:
        g = gcd(g, abs(C))
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Make first non-zero positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _generate_lattice_points(matrix, x_range, y_range):
    """
    Generate points by applying an integer affine transformation (matrix)
    to a rectangular grid defined by x_range and y_range.
    matrix is 2x2 integer matrix.
    Returns (points, valid) where points is (N, 2) array.
    """
    xs = np.arange(x_range[0], x_range[1] + 1)
    ys = np.arange(y_range[0], y_range[1] + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    p_base = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Apply transformation: P' = M @ P
    # M is (2, 2), P is (N, 2). We want (N, 2) @ M.T
    p_trans = (p_base @ matrix.T).astype(np.int64)
    
    # Check for duplicates
    unique_p = np.unique(p_trans, axis=0)
    if len(unique_p) < 54:
        return None, False
        
    # If we have more than 54, we need to select a subset.
    # If we have exactly 54, we are good.
    # For this strategy, we will primarily target transformations that map 
    # a 54-point grid to a 54-point unique set.
    # Common 54-point bases: 1x54, 2x27, 3x18, 6x9, 9x6, 18x3, 27x2, 54x1.
    
    if len(unique_p) == 54:
        return unique_p, True
    else:
        # If we have more, we can't just take the first 54 easily without losing structure.
        # We'll stick to exact matches for now to ensure validity and structure.
        return None, False

def _find_best_lines_for_points(points, time_limit=5.0):
    """
    Use Simulated Annealing to find the best 27 lines for a given set of points.
    """
    start_time = time.time()
    n_points = len(points)
    n_lines = 27
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Precompute all candidate lines from pairs of points
    # This is the "universe" of lines we can choose from.
    line_keys = set()
    # Store mapping from key to (A, B, C) for quick lookup
    key_to_abc = {}
    
    # Optimization: Precompute pair differences?
    # For n=54, pairs = 1431. This is small.
    for i in range(n_points):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n_points):
            xj = x[j]
            yj = y[j]
            
            A = yi - yj
            B = xj - xi
            C = xi * yj - xj * yi
            
            norm = _normalize_line(A, B, C)
            if norm:
                line_keys.add(norm)
                key_to_abc[norm] = norm
                
    # Add some "obvious" lines that might not pass through 2 points?
    # No, any line in the optimal set for a discrete point set 
    # with high incidence likely passes through at least 2 points 
    # unless it's a "franklin" type construction, but with 27 lines 
    # and 54 points, high incidence usually implies structure.
    # However, lines passing through 1 point can be useful if they 
    # don't conflict? No, we want to MAXIMIZE incidences. 
    # A line passing through 1 point contributes 1. 
    # A line passing through 0 points contributes 0.
    # We should only consider lines that pass through at least 1 point.
    # Actually, if we have to pick 27 lines, and there are fewer than 27 
    # lines passing through >= 2 points, we might need to pick lines 
    # passing through 1 point. But typically, grids have many lines with >= 2.
    
    # Let's filter lines to those with count >= 2 first. 
    # If we don't have enough, we'll expand.
    
    candidate_lines = list(line_keys)
    
    # Score each candidate line
    # To speed up SA, we need fast evaluation of a line's incidence count.
    # We can precompute the incidence count for all candidate lines.
    # There are ~1431 candidates. 54 points. 1431 * 54 ops ~ 77k ops. Very fast.
    
    line_counts = {}
    for key in candidate_lines:
        A, B, C = key
        vals = A * x + B * y + C
        count = np.sum(vals == 0)
        line_counts[key] = count
        
    # Sort candidates by count descending to initialize with top 27
    sorted_candidates = sorted(line_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Initialize current set with top 27
    current_set = set([k for k, _ in sorted_candidates[:n_lines]])
    
    # If we don't have 27 lines with count > 0, we might have issues, 
    # but for 54 points, we should have plenty.
    if len(current_set) < n_lines:
        # Pad with random lines from the full candidate list if needed
        # (Though unlikely to be needed for valid grids)
        for k, _ in sorted_candidates[n_lines:]:
            if len(current_set) >= n_lines:
                break
            current_set.add(k)
            
    # Calculate initial score
    # Score is sum of counts of lines in the set?
    # Wait, the objective is sum of incidences.
    # Incidence I = sum_{line in L} count(line).
    # Is it that simple? Yes, I = sum_{p, l} [p on l] = sum_{l} sum_{p} [p on l] = sum_{l} count(l).
    # So we just want to maximize the sum of the counts of the selected lines.
    # This is a "Top-K Selection" problem?
    # If the objective is strictly sum of counts, then the optimal solution 
    # is ALWAYS the top 27 lines by individual count!
    # Why? Because the incidence count of a line depends ONLY on the points, 
    # not on the other lines. The lines do not interact.
    # 
    # RE-READING TASK: "I is the number of (point, line) pairs... proportional to I".
    # Yes, I = sum_{l in L} |{p in P : p on l}|.
    # The lines are independent in the objective function.
    # Therefore, the greedy selection of the top 27 lines by count IS the optimal 
    # solution for a FIXED point set.
    #
    # So Simulated Annealing on the lines is USELESS if the points are fixed.
    # The current program's greedy approach is already optimal for the lines.
    #
    # Therefore, the ONLY way to improve is to change the POINTS.
    # The current program tries a few fixed grids.
    # The new strategy MUST be better at finding good POINT SETS.
    
    # Revised Strategy:
    # 1. Generate diverse point sets (lattices, sheared grids, etc.).
    # 2. For each, compute the optimal lines (top 27).
    # 3. Keep the best.
    #
    # Since line selection is trivial (top 27), we can spend ALL time 
    # exploring point configurations.
    
    return sorted_candidates, current_set

def _evaluate_point_set(points):
    """
    Given a point set, find the optimal lines (top 27) and return the score.
    """
    n_points = len(points)
    n_lines = 27
    x = points[:, 0]
    y = points[:, 1]
    
    # Get all unique lines through pairs
    line_counts = {}
    
    for i in range(n_points):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n_points):
            xj = x[j]
            yj = y[j]
            
            A = yi - yj
            B = xj - xi
            C = xi * yj - xj * yi
            
            norm = _normalize_line(A, B, C)
            if norm:
                # We can update count incrementally? 
                # No, we need the full count for each unique line.
                # Let's just store the key for now.
                if norm not in line_counts:
                    line_counts[norm] = 0 # Placeholder
                    
    # Now count incidences for each unique line
    for key in line_counts.keys():
        A, B, C = key
        vals = A * x + B * y + C
        line_counts[key] = np.sum(vals == 0)
        
    # Sort by count
    sorted_lines = sorted(line_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Take top 27
    if len(sorted_lines) < n_lines:
        # Should not happen for 54 points
        return 0, None, None
        
    best_lines = [k for k, _ in sorted_lines[:n_lines]]
    score = sum(c for _, c in sorted_lines[:n_lines])
    
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    return score, points, lines_arr

def _generate_candidate_points():
    """
    Generate a list of diverse candidate point sets.
    Focus on affine transformations of rectangular grids.
    """
    candidates = []
    
    # Base grids (divisors of 54)
    # (w, h)
    base_grids = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)
    ]
    
    # Affine transformations (2x2 integer matrices with determinant +/- 1 or small)
    # We want to preserve the "grid-like" structure but rotate/shear it.
    # M = [[a, b], [c, d]]
    # Points: (x, y) -> (ax + by, cx + dy)
    
    # Identity
    transforms = [
        np.array([[1, 0], [0, 1]], dtype=np.int64),
        np.array([[1, 1], [0, 1]], dtype=np.int64), # Shear
        np.array([[1, 0], [1, 1]], dtype=np.int64), # Shear
        np.array([[1, 1], [1, 0]], dtype=np.int64), # Rotation/Shear
        np.array([[0, 1], [1, 0]], dtype=np.int64), # Swap
        np.array([[1, -1], [0, 1]], dtype=np.int64),
        np.array([[1, 0], [-1, 1]], dtype=np.int64),
        np.array([[2, 1], [1, 1]], dtype=np.int64), # Non-unimodular? Det = 1.
        np.array([[1, 2], [1, 1]], dtype=np.int64), # Det = -1.
        np.array([[1, 1], [2, 1]], dtype=np.int64), # Det = -1.
        np.array([[3, 1], [1, 1]], dtype=np.int64), # Det = 2. Might create duplicates if not careful.
        np.array([[1, 3], [1, 1]], dtype=np.int64),
    ]
    
    for w, h in base_grids:
        for M in transforms:
            # Generate base grid points
            xs = np.arange(w)
            ys = np.arange(h)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            p_base = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            
            # Apply transform
            p_trans = (p_base @ M.T).astype(np.int64)
            
            # Check uniqueness
            unique_p = np.unique(p_trans, axis=0)
            if len(unique_p) == 54:
                # Center the points to avoid large coordinates? 
                # Not strictly necessary for integer arithmetic, but good practice.
                # Shift to make min 0?
                p_centered = unique_p - np.min(unique_p, axis=0)
                candidates.append(p_centered)
            
    # Also add the standard axis-aligned grids (Identity transform covers this, 
    # but let's ensure we have them explicitly with 1-based indexing if needed, 
    # though 0-based is fine as incidence is translation invariant).
    
    # Add some random "noise" perturbations? 
    # The problem requires integer coordinates. 
    # Random integer points are likely to have low incidence.
    # Structured points are better.
    
    return candidates

def szemeredi_trotter_incidences():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Generate candidates
    point_candidates = _generate_candidate_points()
    
    # Remove duplicates in candidates to save time
    unique_candidates = []
    seen = set()
    for p in point_candidates:
        # Create a hashable representation
        # Sort points to ensure canonical form for comparison
        p_sorted = np.sort(p, axis=0)
        key = tuple(map(tuple, p_sorted))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)
            
    # Evaluate each candidate
    for p in unique_candidates:
        if time.time() + safety_margin > start_time + deadline:
            break
            
        try:
            score, pts, lines = _evaluate_point_set(p)
            
            if score > best_score:
                best_score = score
                best_points = pts
                best_lines = lines
                
        except Exception as e:
            continue
            
    # If no valid candidate found (should not happen), fallback
    if best_points is None:
        # Fallback to 6x9 grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate lines for fallback
        x = best_points[:, 0]
        y = best_points[:, 1]
        line_counts = {}
        for i in range(len(best_points)):
            for j in range(i + 1, len(best_points)):
                A = y[i] - y[j]
                B = x[j] - x[i]
                C = x[i] * y[j] - x[j] * y[i]
                norm = _normalize_line(A, B, C)
                if norm:
                    if norm not in line_counts:
                        line_counts[norm] = 0
        for key in line_counts.keys():
            A, B, C = key
            vals = A * x + B * y + C
            line_counts[key] = np.sum(vals == 0)
            
        sorted_lines = sorted(line_counts.items(), key=lambda item: item[1], reverse=True)
        best_lines = np.array([k for k, _ in sorted_lines[:27]], dtype=np.int64)
        
    return best_points, best_lines
