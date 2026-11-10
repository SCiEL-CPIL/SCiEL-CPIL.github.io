# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def _count_incidences(points, lines):
    """Count incidences: number of (point, line) pairs where point lies on line."""
    # points: (n, 2), lines: (m, 3) with A*x + B*y + C = 0
    # For each line, check all points
    # Vectorized: for line (A, B, C), check A*px + B*py + C == 0
    total = 0
    for i in range(lines.shape[0]):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.count_nonzero(vals == 0)
    return total


def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    from math import gcd
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_two_points(p1, p2):
    """Get line through two points (x1,y1) and (x2,y2) in normalized form."""
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = -(A * p1[0] + B * p1[1])
    return _normalize_line(A, B, C)


def _get_all_lines(points):
    """Get all distinct lines determined by pairs of points."""
    n = points.shape[0]
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            ln = _line_from_two_points(tuple(points[i]), tuple(points[j]))
            line_set.add(ln)
    return list(line_set)


def _line_incidence_count(line, points):
    """Count how many points lie on the given line."""
    A, B, C = line
    vals = A * points[:, 0] + B * points[:, 1] + C
    return np.count_nonzero(vals == 0)


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Strategy: Use a 7x8 grid (56 points), remove 2 points to get 54,
    then select the 27 best lines (by incidence count) that are distinct.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Best result so far (the baseline grid construction)
    k = 3
    xs_base = np.arange(1, k + 1)
    ys_base = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs_base, ys_base, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    base_lines = np.array(rows, dtype=np.int64)
    best_score = _count_incidences(base_points, base_lines)
    best_points = base_points.copy()
    best_lines = base_lines.copy()
    
    # 7x8 grid: 56 points
    W, H = 7, 8
    xs = np.arange(0, W)
    ys = np.arange(0, H)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    full_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_full = full_points.shape[0]  # 56
    
    # We need to remove 2 points from 56 to get 54
    # Try all pairs of points to remove: C(56,2) = 1540
    # For each, compute all lines and pick top 27
    
    # Precompute: for each pair of points, the line they define
    # But we need to be smart about this
    
    # Strategy: For each pair of points to remove, compute the incidence
    # of each possible line with the remaining 54 points, then pick top 27.
    
    # To speed up: precompute for each pair of indices (i,j) the line
    # and for each line, which points it passes through.
    
    # Let's precompute all lines from the 56-point grid
    all_lines_dict = {}  # line -> set of point indices
    for i in range(n_full):
        for j in range(i + 1, n_full):
            ln = _line_from_two_points(tuple(full_points[i]), tuple(full_points[j]))
            if ln not in all_lines_dict:
                all_lines_dict[ln] = set()
            all_lines_dict[ln].add(i)
            all_lines_dict[ln].add(j)
    
    # For each line, store the set of points it contains
    line_point_sets = list(all_lines_dict.items())  # list of (line, set_of_indices)
    
    # Now try removing pairs of points
    best_found = best_score
    
    # Sort removal pairs by some heuristic? Or just iterate all.
    # 1540 pairs, each requiring: for each line, compute incidence with remaining 54 points.
    # Number of lines in 7x8 grid: let's estimate. 
    # For a W x H grid, the number of distinct lines is roughly O(W*H*(W+H))
    # For 7x8, it's manageable.
    
    num_lines = len(line_point_sets)
    
    for remove_i in range(n_full):
        for remove_j in range(remove_i + 1, n_full):
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:  # 30 second safety margin
                break
            
            # The remaining points are all except remove_i and remove_j
            # For each line, incidence = |points_on_line| - (1 if remove_i in line) - (1 if remove_j in line)
            # But we need to be careful: the line must still have at least 2 points to be valid?
            # Actually no - a line is defined by its equation, not by points. But for it to be
            # "from the grid", it needs at least 2 grid points. After removal, it might have fewer.
            # But the task just says we need 27 lines - they don't have to pass through 2+ of our points.
            # However, lines that pass through 0 of our points contribute 0, so they're useless.
            
            # Compute incidence for each line
            # line_incidence = len(line_points) - indicator(remove_i in line) - indicator(remove_j in line)
            
            # We want the top 27 lines by incidence
            incidences = np.zeros(num_lines, dtype=np.int64)
            for idx, (ln, pts) in enumerate(line_point_sets):
                cnt = len(pts)
                if remove_i in pts:
                    cnt -= 1
                if remove_j in pts:
                    cnt -= 1
                incidences[idx] = cnt
            
            # Get top 27
            if num_lines >= 27:
                # Get indices of top 27
                # Use argpartition for efficiency
                top_idx = np.argpartition(incidences, -27)[-27:]
                top_inc = incidences[top_idx]
                total_inc = np.sum(top_inc)
                
                if total_inc > best_found:
                    best_found = int(total_inc)
                    # Store the best lines
                    best_lines_for_this = np.array([line_point_sets[i][0] for i in top_idx], dtype=np.int64)
                    # Remove the two points
                    mask = np.ones(n_full, dtype=bool)
                    mask[remove_i] = False
                    mask[remove_j] = False
                    best_points_for_this = full_points[mask].copy()
                    
                    # Update best
                    best_score = best_found
                    best_points = best_points_for_this
                    best_lines = best_lines_for_this
            else:
                # All lines
                total_inc = np.sum(incidences)
                if total_inc > best_found:
                    best_found = int(total_inc)
                    best_lines_for_this = np.array([line_point_sets[i][0] for i in range(num_lines)], dtype=np.int64)
                    mask = np.ones(n_full, dtype=bool)
                    mask[remove_i] = False
                    mask[remove_j] = False
                    best_points_for_this = full_points[mask].copy()
                    
                    best_score = best_found
                    best_points = best_points_for_this
                    best_lines = best_lines_for_this
        
        # Check time
        if time.time() - start_time > deadline - 30:
            break
    
    # Verify the best result
    if best_points is not None and best_lines is not None:
        actual_score = _count_incidences(best_points, best_lines)
        if actual_score >= best_score:
            return best_points.astype(np.int64), best_lines.astype(np.int64)
    
    # Fallback to baseline
    return base_points.astype(np.int64), base_lines.astype(np.int64)


# EVOLVE-BLOCK-END
