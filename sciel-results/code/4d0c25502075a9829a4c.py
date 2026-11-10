# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd


def _gcd3(a, b, c):
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    g = _gcd3(A, B, C)
    if g == 0:
        return (0, 0, 0)
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
    """Count (point, line) incidences exactly using int64."""
    vals = lines[:, 0][None, :] * points[:, 0:1] + lines[:, 1][None, :] * points[:, 1:2] + lines[:, 2][None, :]
    return int(np.sum(vals == 0))


def _construct_clique_star():
    """
    Clique-star hybrid: 27 points on vertical line x=0, 27 points on 3x9 grid.
    Then select best 27 lines.
    """
    # 27 points on vertical line x=0, y=0..26
    star_points = np.array([(0, y) for y in range(27)], dtype=np.int64)
    
    # 27 points on 3x9 grid, offset to avoid overlap with star
    grid_points = []
    for i in range(3):
        for j in range(9):
            grid_points.append((i + 1, j))
    grid_points = np.array(grid_points, dtype=np.int64)
    
    points = np.vstack([star_points, grid_points])
    
    # Generate candidate lines
    candidate_lines = set()
    
    # The vertical line through all star points
    candidate_lines.add(_normalize_line(1, 0, 0))
    
    # Lines through pairs of points
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            candidate_lines.add(_normalize_line(A, B, C))
    
    # Add some extra candidate lines (small coefficients)
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 30):
                candidate_lines.add(_normalize_line(A, B, C))
    
    # Score each candidate line by incidence count
    line_scores = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(vals == 0))
        if cnt >= 2:  # Only keep lines with at least 2 incidences
            line_scores[nl] = cnt
    
    # Sort by score descending
    sorted_lines = sorted(line_scores.items(), key=lambda x: (-x[1], x[0]))
    
    # Greedily select 27 lines
    selected = []
    selected_set = set()
    for nl, cnt in sorted_lines:
        if nl in selected_set:
            continue
        selected.append(nl)
        selected_set.add(nl)
        if len(selected) == 27:
            break
    
    # Pad if needed
    if len(selected) < 27:
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
    return points, lines_arr


def _optimize_lines_for_points(points, max_time):
    """
    Given fixed points, find best 27 lines via local search.
    """
    n = points.shape[0]
    deadline = time.time() + max_time
    
    # Generate all candidate lines from point pairs
    candidate_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            candidate_lines.add(_normalize_line(A, B, C))
    
    # Add small-coefficient lines
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0:
                continue
            for C in range(-40, 40):
                candidate_lines.add(_normalize_line(A, B, C))
    
    # Score each candidate line
    line_scores = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(vals == 0))
        if cnt >= 2:
            line_scores[nl] = cnt
    
    # Sort by score descending
    sorted_lines = sorted(line_scores.items(), key=lambda x: (-x[1], x[0]))
    
    # Greedily select 27 lines
    selected = []
    selected_set = set()
    for nl, cnt in sorted_lines:
        if nl in selected_set:
            continue
        selected.append(nl)
        selected_set.add(nl)
        if len(selected) == 27:
            break
    
    # Pad if needed
    if len(selected) < 27:
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
    current_count = _count_incidents(points, lines_arr)
    
    # Local search: try swapping one selected line with a non-selected candidate
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


def _hill_climb_points(points, lines, max_time):
    """
    Hill-climb by moving individual points to maximize incidences.
    """
    best_points = points.copy()
    best_count = _count_incidents(best_points, lines)
    deadline = time.time() + max_time
    rng = np.random.default_rng(12345)
    
    n = 54
    m = 27
    
    A_arr = lines[:, 0].astype(np.float64)
    B_arr = lines[:, 1].astype(np.float64)
    C_arr = lines[:, 2].astype(np.float64)
    
    x_candidates = np.arange(-15, 25, dtype=np.int64)
    y_candidates = np.arange(-15, 35, dtype=np.int64)
    
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            orig_x, orig_y = int(best_points[i, 0]), int(best_points[i, 1])
            
            # Try several random candidate positions
            for trial in range(10):
                cx = int(x_candidates[rng.integers(len(x_candidates))])
                cy = int(y_candidates[rng.integers(len(y_candidates))])
                
                # Check if this position is already taken
                if any(best_points[j, 0] == cx and best_points[j, 1] == cy for j in range(n) if j != i):
                    continue
                
                new_on = np.sum(A_arr * cx + B_arr * cy + C_arr == 0)
                old_on = np.sum(A_arr * orig_x + B_arr * orig_y + C_arr == 0)
                
                if new_on > old_on:
                    best_points[i, 0] = cx
                    best_points[i, 1] = cy
                    best_count += (new_on - old_on)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
    
    return best_points, best_count


def _validate(points, lines):
    """Check all validity constraints. Return True if valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.allclose(points, np.round(points), atol=1e-6) or not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    pt_set = set()
    for p in points:
        t = (int(p[0]), int(p[1]))
        if t in pt_set:
            return False
        pt_set.add(t)
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


def _construct_grid_diagonal():
    """Fallback: 6x9 grid with 15 grid lines + 12 best diagonals."""
    points_x = np.arange(6)
    points_y = np.arange(9)
    px, py = np.meshgrid(points_x, points_y, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for i in range(6):
        lines.append((1, 0, -i))
    for j in range(9):
        lines.append((0, 1, -j))
    c_vals = [0, -1, -2, -3, 1, -4, 2, -5, 3, -6, 4, -7]
    for c in c_vals:
        lines.append((1, -1, c))
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _try_multiple_start_configs(points_list, lines_list, total_budget, start_time):
    """
    Try multiple starting configurations and return the best result.
    """
    best_points = None
    best_lines = None
    best_count = -1
    
    for idx, (pts, lns) in enumerate(zip(points_list, lines_list)):
        elapsed = time.time() - start_time
        if elapsed > total_budget - 60:
            break
        
        count = _count_incidents(pts, lns)
        
        # Optimize lines for these points
        if time.time() - start_time < total_budget - 30:
            remaining_for_lines = min(60, total_budget - (time.time() - start_time) - 30)
            if remaining_for_lines > 10:
                lines_opt, count_opt = _optimize_lines_for_points(pts, max_time=remaining_for_lines)
                if count_opt > count:
                    lns = lines_opt
                    count = count_opt
        
        # Hill-climb points
        if time.time() - start_time < total_budget - 10:
            remaining_for_hc = min(40, total_budget - (time.time() - start_time) - 10)
            if remaining_for_hc > 10:
                pts_hc, count_hc = _hill_climb_points(pts, lns, max_time=remaining_for_hc)
                if count_hc > count:
                    pts = pts_hc
                    count = count_hc
        
        # Re-optimize lines for new points
        if time.time() - start_time < total_budget - 10:
            remaining_for_lines2 = min(30, total_budget - (time.time() - start_time) - 10)
            if remaining_for_lines2 > 5:
                lines_opt2, count_opt2 = _optimize_lines_for_points(pts, max_time=remaining_for_lines2)
                if count_opt2 > count:
                    lns = lines_opt2
                    count = count_opt2
        
        if count > best_count:
            best_count = count
            best_points = pts.copy()
            best_lines = lns.copy()
    
    return best_points, best_lines, best_count


def szemeredi_trotter_incidences():
    start_time = time.time()
    total_budget = 580  # seconds, with 50s safety margin
    
    # Generate multiple starting configurations
    start_configs = []
    
    # Config 1: Clique-star hybrid
    pts1, lns1 = _construct_clique_star()
    start_configs.append((pts1, lns1))
    
    # Config 2: Standard 6x9 grid
    pts2, lns2 = _construct_grid_diagonal()
    start_configs.append((pts2, lns2))
    
    # Config 3: 3x9 grid with shifted star
    star_points = np.array([(0, y) for y in range(27)], dtype=np.int64)
    grid_points = []
    for i in range(3):
        for j in range(9):
            grid_points.append((i + 2, j + 1))
    grid_points = np.array(grid_points, dtype=np.int64)
    pts3 = np.vstack([star_points, grid_points])
    lns3 = _optimize_lines_for_points(pts3, max_time=30)[0]
    start_configs.append((pts3, lns3))
    
    # Config 4: 2x13.5 not possible, try 2x14 grid + 26 star
    star_points4 = np.array([(0, y) for y in range(26)], dtype=np.int64)
    grid_points4 = []
    for i in range(2):
        for j in range(14):
            grid_points4.append((i + 1, j))
    grid_points4 = np.array(grid_points4, dtype=np.int64)
    # Need 54 points total: 26 + 28 = 54, but 2x14=28
    pts4 = np.vstack([star_points4, grid_points4])
    lns4 = _optimize_lines_for_points(pts4, max_time=30)[0]
    start_configs.append((pts4, lns4))
    
    # Config 5: 9x3 grid (transposed)
    pts5, lns5 = _construct_grid_diagonal()
    # Transpose the grid
    pts5_new = np.column_stack([pts5[:, 1], pts5[:, 0]])
    # Re-optimize lines
    lns5_new = _optimize_lines_for_points(pts5_new, max_time=30)[0]
    start_configs.append((pts5_new, lns5_new))
    
    # Try all configs
    best_points, best_lines, best_count = _try_multiple_start_configs(
        [c[0] for c in start_configs],
        [c[1] for c in start_configs],
        total_budget, start_time
    )
    
    # If no good result, fall back to safe construction
    if best_points is None or not _validate(best_points, best_lines):
        best_points, best_lines = _construct_grid_diagonal()
    
    # Final validation
    if not _validate(best_points, best_lines):
        best_points, best_lines = _construct_grid_diagonal()
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
