# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from math import gcd


def _gcd(a, b, c):
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    g = _gcd(A, B, C)
    A //= g
    B //= g
    C //= g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidents(points, lines):
    vals = lines[:, 0][None, :] * points[:, 0:1] + lines[:, 1][None, :] * points[:, 1:2] + lines[:, 2][None, :]
    return int(np.sum(vals == 0))


def _construct_grid_diagonal():
    """6x9 grid with 15 grid lines + 12 best diagonals. Gives 160 incidences."""
    points_x = np.arange(6)
    points_y = np.arange(9)
    px, py = np.meshgrid(points_x, points_y, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for i in range(6):
        lines.append((1, 0, -i))
    for j in range(9):
        lines.append((0, 1, -j))
    
    # Generate candidates for diagonals
    candidates = []
    # x - y = c
    for c in range(-10, 10):
        candidates.append((1, -1, c))
    # x + y = c
    for c in range(-10, 20):
        candidates.append((1, 1, c))
    # Small slopes
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0: continue
            if A == 1 and B == -1: continue # already added
            if A == 1 and B == 1: continue
            for C in range(-20, 20):
                candidates.append((A, B, C))
                
    # Normalize and unique
    unique_lines = set()
    for l in candidates:
        unique_lines.add(_normalize_line(*l))
        
    # Score each candidate line
    scored_lines = []
    for l in unique_lines:
        A, B, C = l
        vals = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(vals == 0))
        if cnt > 0:
            scored_lines.append((cnt, l))
            
    # Sort by count desc
    scored_lines.sort(key=lambda x: -x[0])
    
    # Pick top 12 that are not in the grid lines
    grid_set = set([(1,0,-i) for i in range(6)] + [(0,1,-j) for j in range(9)])
    # Note: normalized grid lines might be different?
    # (1,0,-i) is normalized if i>=0.
    # (0,1,-j) is normalized if j>=0.
    selected_diags = []
    for cnt, l in scored_lines:
        if l not in grid_set and len(selected_diags) < 12:
            selected_diags.append(l)
            
    for l in selected_diags:
        lines.append(l)
        
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _validate(points, lines):
    """Check all validity constraints. Return True if valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.allclose(points, np.round(points), atol=1e-6) or not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    # Repeated points
    pt_set = set()
    for p in points:
        t = (int(p[0]), int(p[1]))
        if t in pt_set:
            return False
        pt_set.add(t)
    # Lines
    line_set = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        nl = _normalize_line(A, B, C)
        if nl in line_set:
            return False
        line_set.add(nl)
    return True


def _hill_climb(points, lines, max_time):
    """
    Hill-climb by moving individual points to maximize incidences.
    Returns (best_points, best_count).
    """
    best_points = points.copy()
    best_count = _count_incidents(best_points, lines)
    deadline = time.time() + max_time
    rng = np.random.default_rng(42)
    
    # Precompute for efficiency
    n = 54
    m = 27
    
    # For each line, precompute A, B, C as float for fast evaluation
    A_arr = lines[:, 0].astype(np.float64)
    B_arr = lines[:, 1].astype(np.float64)
    C_arr = lines[:, 2].astype(np.float64)
    
    # Candidate x, y values for moved points
    x_candidates = np.arange(-10, 20, dtype=np.int64)
    y_candidates = np.arange(-10, 20, dtype=np.int64)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        # Try moving each point to a random candidate position
        for i in range(n):
            if time.time() > deadline - 5:
                break
            orig_x, orig_y = int(best_points[i, 0]), int(best_points[i, 1])
            
            # Try several random candidate positions
            for trial in range(8):
                cx = int(x_candidates[rng.integers(len(x_candidates))])
                cy = int(y_candidates[rng.integers(len(y_candidates))])
                
                # Check if this position is already taken by another point
                if any(best_points[j, 0] == cx and best_points[j, 1] == cy for j in range(n) if j != i):
                    continue
                
                # Evaluate new incidence count
                # Only need to check lines affected by point i
                # A point on line j iff A_j*cx + B_j*cy + C_j == 0
                new_on = np.sum(A_arr * cx + B_arr * cy + C_arr == 0)
                
                # Old incidences from point i
                old_on = np.sum(A_arr * orig_x + B_arr * orig_y + C_arr == 0)
                
                if new_on > old_on:
                    best_points[i, 0] = cx
                    best_points[i, 1] = cy
                    best_count += (new_on - old_on)
                    improved = True
                    no_improve = 0
                    break  # Move to next point
        
        if not improved:
            no_improve += 1
            # Try a few more random restarts of the point order
            if no_improve >= max_no_improve:
                break
    
    return best_points, best_count


def _optimize_lines_for_points(points, max_time):
    """
    Given fixed points, try to find better lines via local search.
    Lines are defined by pairs of points (or single point + direction).
    We search over the set of all lines determined by point pairs.
    """
    n = points.shape[0]
    deadline = time.time() + max_time
    
    # Generate all candidate lines from point pairs
    # Line through points (x1,y1) and (x2,y2): A = y2-y1, B = x1-x2, C = x2*y1 - x1*y2
    candidate_lines = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            nl = _normalize_line(A, B, C)
            candidate_lines.append(nl)
    
    # Remove duplicates, keep track of incidence count
    line_inc = {}
    for nl in candidate_lines:
        if nl not in line_inc:
            A, B, C = nl
            # Count incidences
            vals = A * points[:, 0] + B * points[:, 1] + C
            cnt = int(np.sum(vals == 0))
            line_inc[nl] = cnt
    
    # Sort by incidence count descending
    sorted_lines = sorted(line_inc.items(), key=lambda x: -x[1])
    
    # Greedily select 27 distinct lines
    selected = []
    selected_set = set()
    for nl, cnt in sorted_lines:
        if nl in selected_set:
            continue
        selected.append(nl)
        selected_set.add(nl)
        if len(selected) == 27:
            break
    
    if len(selected) < 27:
        # Pad with arbitrary lines
        for c in range(-100, 100):
            nl = _normalize_line(0, 1, c)
            if nl not in selected_set:
                selected.append(nl)
                selected_set.add(nl)
                if len(selected) == 27:
                    break
        if len(selected) < 27:
            for c in range(-100, 100):
                nl = _normalize_line(1, 0, c)
                if nl not in selected_set:
                    selected.append(nl)
                    selected_set.add(nl)
                    if len(selected) == 27:
                        break
    
    lines_arr = np.array(selected, dtype=np.int64)
    count = _count_incidents(points, lines_arr)
    
    # Local search: try swapping one selected line with a non-selected candidate
    current_count = count
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_lines = [tuple(l) for l in lines_arr]
        current_set = set(current_lines)
        
        # Compute current per-line incidence
        line_counts = {}
        for nl in current_lines:
            A, B, C = nl
            vals = A * points[:, 0] + B * points[:, 1] + C
            line_counts[nl] = int(np.sum(vals == 0))
        
        # Find the worst line
        worst_idx = 0
        worst_cnt = line_counts[current_lines[0]]
        for idx, nl in enumerate(current_lines):
            if line_counts[nl] < worst_cnt:
                worst_cnt = line_counts[nl]
                worst_idx = idx
        
        # Try replacing worst line with a better non-selected line
        best_gain = 0
        best_replacement = None
        for nl, cnt in sorted_lines:
            if nl in current_set:
                continue
            gain = cnt - worst_cnt
            if gain > best_gain:
                best_gain = gain
                best_replacement = nl
        
        if best_replacement is not None and best_gain > 0:
            current_lines[worst_idx] = best_replacement
            current_set.discard(current_lines[worst_idx])
            current_set.add(best_replacement)
            lines_arr = np.array(current_lines, dtype=np.int64)
            current_count += best_gain
            improved = True
    
    return lines_arr, current_count


def szemeredi_trotter_incidences():
    start_time = time.time()
    total_budget = 580  # seconds, with 50s safety margin
    
    # Start with the known-good construction
    points, lines = _construct_grid_diagonal()
    best_count = _count_incidents(points, lines)
    
    # Phase 1: Optimize lines for the grid points
    if time.time() - start_time < total_budget - 30:
        lines_opt, count_opt = _optimize_lines_for_points(points, max_time=120)
        if count_opt > best_count:
            lines = lines_opt
            best_count = count_opt
    
    # Phase 2: Hill-climb points with optimized lines
    if time.time() - start_time < total_budget - 10:
        remaining = total_budget - (time.time() - start_time) - 5
        if remaining > 30:
            points_hc, count_hc = _hill_climb(points, lines, max_time=remaining)
            if count_hc > best_count:
                points = points_hc
                best_count = count_hc
                # Re-optimize lines for new points
                if time.time() - start_time < total_budget - 10:
                    lines_opt2, count_opt2 = _optimize_lines_for_points(points, max_time=60)
                    if count_opt2 > best_count:
                        lines = lines_opt2
                        best_count = count_opt2
    
    # Phase 3: Another round of hill-climbing with the best lines found
    if time.time() - start_time < total_budget - 10:
        remaining = total_budget - (time.time() - start_time) - 5
        if remaining > 30:
            points_hc2, count_hc2 = _hill_climb(points, lines, max_time=remaining)
            if count_hc2 > best_count:
                points = points_hc2
                best_count = count_hc2
    
    # Final validation
    if not _validate(points, lines):
        # Fall back to the safe construction
        points, lines = _construct_grid_diagonal()
    
    return points, lines


# EVOLVE-BLOCK-END
