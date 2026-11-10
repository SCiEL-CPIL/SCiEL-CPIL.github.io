# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def _normalize_line(A, B, C):
    """Normalize a line (A,B,C) so that first nonzero is positive and gcd=1."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A != 0:
        if A < 0:
            A, B, C = -A, -B, -C
    elif B != 0:
        if B < 0:
            A, B, C = -A, -B, -C
    else:
        if C < 0:
            C = -C
    return (A, B, C)


def _incidence_count(points, A, B, C):
    """Count points on line Ax + By + C = 0."""
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))


def _total_incidence(points, lines):
    """Total incidences between points and lines."""
    total = 0
    for i in range(len(lines)):
        A, B, C = lines[i]
        total += _incidence_count(points, A, B, C)
    return total


def _generate_candidate_lines(points, max_coef=4):
    """Generate candidate lines from pairs of points and small-coefficient lines."""
    n = len(points)
    candidates = {}  # normalized -> (A, B, C, count)
    
    # Lines through pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm not in candidates:
                cnt = _incidence_count(points, A, B, C)
                candidates[norm] = (A, B, C, cnt)
    
    # Small-coefficient lines
    for A in range(-max_coef, max_coef + 1):
        for B in range(-max_coef, max_coef + 1):
            if A == 0 and B == 0:
                continue
            # Determine range of C
            # A*x + B*y + C = 0 => C = -(A*x + B*y)
            # x, y are in the range of the points
            min_x, max_x = points[:, 0].min(), points[:, 0].max()
            min_y, max_y = points[:, 1].min(), points[:, 1].max()
            
            if A > 0:
                c_min = -(A * max_x + B * max_y) if B > 0 else -(A * max_x + B * min_y)
                c_max = -(A * min_x + B * min_y) if B > 0 else -(A * min_x + B * max_y)
            elif A < 0:
                c_min = -(A * min_x + B * min_y) if B > 0 else -(A * min_x + B * max_y)
                c_max = -(A * max_x + B * max_y) if B > 0 else -(A * max_x + B * min_y)
            else:  # A == 0, B != 0
                if B > 0:
                    c_min = -(B * max_y)
                    c_max = -(B * min_y)
                else:
                    c_min = -(B * min_y)
                    c_max = -(B * max_y)
            
            for C in range(int(c_min), int(c_max) + 1):
                A_norm, B_norm, C_norm = _normalize_line(A, B, C)
                norm = (A_norm, B_norm, C_norm)
                if norm not in candidates:
                    cnt = _incidence_count(points, A, B, C)
                    candidates[norm] = (A, B, C, cnt)
    
    return candidates


def _greedy_select(points, candidates, k=27):
    """Greedy selection of k lines maximizing total incidences."""
    # Sort candidates by count descending
    sorted_cands = sorted(candidates.values(), key=lambda x: -x[3])
    
    selected = []
    used_norms = set()
    total = 0
    
    for A, B, C, cnt in sorted_cands:
        if len(selected) >= k:
            break
        if cnt <= 0:
            break
        norm = _normalize_line(A, B, C)
        if norm in used_norms:
            continue
        
        # Check marginal gain (simple: just add the count, ignoring overlaps)
        selected.append((A, B, C))
        used_norms.add(norm)
        total += cnt
    
    # Fill remaining slots if needed
    if len(selected) < k:
        for A, B, C, cnt in sorted_cands:
            if len(selected) >= k:
                break
            norm = _normalize_line(A, B, C)
            if norm not in used_norms:
                selected.append((A, B, C))
                used_norms.add(norm)
    
    return selected[:k]


def _local_search(points, lines, candidates, max_iter=500, deadline=None):
    """Local search: try swapping one line with another to improve total."""
    lines = [tuple(l) for l in lines]
    n_lines = len(lines)
    n_points = len(points)
    
    # Precompute incidence mask for each line
    def line_mask(A, B, C):
        vals = A * points[:, 0] + B * points[:, 1] + C
        return (vals == 0)
    
    current_masks = [line_mask(A, B, C) for A, B, C in lines]
    current_total = sum(np.sum(m) for m in current_masks)
    
    # Build candidate list as (A, B, C, norm, mask)
    cand_list = []
    for norm, (A, B, C, cnt) in candidates.items():
        if cnt > 0:
            mask = line_mask(A, B, C)
            cand_list.append((A, B, C, norm, mask))
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if deadline and time.time() > deadline:
            break
        
        for i in range(n_lines):
            A_i, B_i, C_i = lines[i]
            norm_i = _normalize_line(A_i, B_i, C_i)
            mask_i = current_masks[i]
            
            best_gain = 0
            best_cand = None
            
            for j, (A_j, B_j, C_j, norm_j, mask_j) in enumerate(cand_list):
                if norm_j == norm_i:
                    continue
                # Check if this candidate is already in lines
                already = False
                for k in range(n_lines):
                    if _normalize_line(lines[k][0], lines[k][1], lines[k][2]) == norm_j:
                        already = True
                        break
                if already:
                    continue
                
                # Marginal gain: points on new line not on old line, minus points on old line not on new line
                gain = np.sum(mask_j & ~mask_i) - np.sum(mask_i & ~mask_j)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = j
            
            if best_cand is not None:
                A_new, B_new, C_new, norm_new, mask_new = cand_list[best_cand]
                lines[i] = (A_new, B_new, C_new)
                current_masks[i] = mask_new
                current_total += best_gain
                improved = True
    
    return lines, current_total


def _construct_grid(points_shape):
    """Construct a grid of given shape."""
    rows, cols = points_shape
    xs = np.arange(1, cols + 1)
    ys = np.arange(1, rows + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _try_grid_and_optimize(rows, cols, deadline):
    """Try a grid construction and optimize line selection."""
    points = _construct_grid((rows, cols))
    n_points = len(points)
    
    # Generate candidates
    candidates = _generate_candidate_lines(points, max_coef=3)
    
    # Greedy initial selection
    selected = _greedy_select(points, candidates, k=27)
    lines = np.array(selected, dtype=np.int64)
    total = _total_incidence(points, lines)
    
    # Local search
    if time.time() < deadline:
        lines, total = _local_search(points, lines, candidates, max_iter=200, deadline=deadline)
    
    return points, lines, total


def _try_sheared_grid(rows, cols, shear, deadline):
    """Try a sheared grid."""
    xs = np.arange(1, cols + 1)
    ys = np.arange(1, rows + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    # Apply shear: x' = x + shear*y
    px = px + shear * py
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Check for duplicate points
    if len(np.unique(points, axis=0)) != len(points):
        return None, None, -1
    
    candidates = _generate_candidate_lines(points, max_coef=3)
    selected = _greedy_select(points, candidates, k=27)
    lines = np.array(selected, dtype=np.int64)
    total = _total_incidence(points, lines)
    
    if time.time() < deadline:
        lines, total = _local_search(points, lines, candidates, max_iter=100, deadline=deadline)
    
    return points, lines, total


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 570
    
    best_points = None
    best_lines = None
    best_total = 0
    
    # Try various grid configurations
    grid_configs = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2),
        (5, 11), (11, 5), (4, 14), (14, 4),
    ]
    
    for rows, cols in grid_configs:
        if time.time() > deadline - 60:
            break
        try:
            points, lines, total = _try_grid_and_optimize(rows, cols, deadline)
            if total > best_total:
                best_points = points.copy()
                best_lines = lines.copy()
                best_total = total
        except Exception:
            continue
    
    # Try sheared grids
    for rows, cols in [(6, 9), (9, 6), (3, 18), (18, 3)]:
        if time.time() > deadline - 30:
            break
        for shear in [-2, -1, 1, 2]:
            if time.time() > deadline - 30:
                break
            try:
                points, lines, total = _try_sheared_grid(rows, cols, shear, deadline)
                if points is not None and total > best_total:
                    best_points = points.copy()
                    best_lines = lines.copy()
                    best_total = total
            except Exception:
                continue
    
    # Fallback to known good construction if nothing found
    if best_points is None:
        points = _construct_grid((6, 9))
        lines_list = []
        for c in range(1, 7):
            lines_list.append([1, 0, -c])
        for c in range(1, 10):
            lines_list.append([0, 1, -c])
        for b in [-1, 0, 1, 2, 3, 4]:
            lines_list.append([1, -1, b])
        for b in [7, 8, 9, 10, 6, 11]:
            lines_list.append([1, 1, -b])
        best_points = points
        best_lines = np.array(lines_list, dtype=np.int64)
    
    # Ensure we have exactly 27 lines
    if len(best_lines) < 27:
        # Pad with low-incidence lines
        candidates = _generate_candidate_lines(best_points, max_coef=2)
        used_norms = set()
        for l in best_lines:
            used_norms.add(_normalize_line(int(l[0]), int(l[1]), int(l[2])))
        
        for norm, (A, B, C, cnt) in sorted(candidates.items(), key=lambda x: -x[1][3]):
            if len(best_lines) >= 27:
                break
            if norm not in used_norms:
                best_lines = np.vstack([best_lines, [[A, B, C]]])
                used_norms.add(norm)
    
    best_lines = best_lines[:27].astype(np.int64)
    best_points = best_points.astype(np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
