# EVOLVE-BLOCK-START
import numpy as np
import time
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
        key = _normalize_line(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True


def _count_incidences(points, lines):
    x = points[:, 0:1].astype(np.int64)
    y = points[:, 1:2].astype(np.int64)
    A = lines[:, 0:1].astype(np.int64)
    B = lines[:, 1:2].astype(np.int64)
    C = lines[:, 2:3].astype(np.int64)
    vals = x * A.T + y * B.T + C.T
    return int(np.sum(vals == 0))


def _line_from_two_points(p1, p2):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    key = _normalize_line(A, B, C)
    if key[0] == 0 and key[1] == 0:
        return None
    return key


def _point_from_two_lines(l1, l2):
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # parallel
    x = (B1 * C2 - B2 * C1) // det if (B1 * C2 - B2 * C1) % det == 0 else None
    y = (C1 * A2 - C2 * A1) // det if (C1 * A2 - C2 * A1) % det == 0 else None
    if x is None or y is None:
        return None
    return (x, y)


def _score_line_set(lines_list):
    """Given a set of lines, compute max incidences by placing 54 points optimally.
    We place points at intersections of pairs of lines. Count how many lines each intersection lies on.
    Pick top 54 unique intersections (or fill with points on single lines if needed)."""
    n = len(lines_list)
    # Compute all pairwise intersections and their line counts
    intersections = {}  # point -> set of line indices
    for i in range(n):
        for j in range(i + 1, n):
            pt = _point_from_two_lines(lines_list[i], lines_list[j])
            if pt is None:
                continue
            if pt not in intersections:
                intersections[pt] = set()
            intersections[pt].add(i)
            intersections[pt].add(j)
    
    # Also, a point on a line that isn't an intersection of two selected lines
    # can still be incident to that one line. But for maximizing, intersections are best.
    
    # Sort intersections by number of lines they're on
    sorted_pts = sorted(intersections.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    if len(sorted_pts) < 54:
        # Need to add points that lie on exactly 1 or 0 selected lines
        # This is a fallback; try to add points on existing lines
        existing_pts = set(pt for pt, _ in sorted_pts)
        # Try adding points along each line
        for i in range(n):
            A, B, C = lines_list[i]
            # Find integer points on this line not already used
            # Try a range of parameter values
            if A != 0:
                # x is free, y = -(A*x + C)/B if B != 0
                if B != 0:
                    for x_try in range(-10, 11):
                        num = -(A * x_try + C)
                        if num % B == 0:
                            y_try = num // B
                            pt = (x_try, y_try)
                            if pt not in existing_pts:
                                intersections[pt] = {i}
                                existing_pts.add(pt)
                else:
                    # A*x + C = 0, x fixed
                    if C % A == 0:
                        x_try = -C // A
                        for y_try in range(-10, 11):
                            pt = (x_try, y_try)
                            if pt not in existing_pts:
                                intersections[pt] = {i}
                                existing_pts.add(pt)
            else:
                # B*y + C = 0, y fixed
                if C % B == 0:
                    y_try = -C // B
                    for x_try in range(-10, 11):
                        pt = (x_try, y_try)
                        if pt not in existing_pts:
                            intersections[pt] = {i}
                            existing_pts.add(pt)
        
        # Re-sort
        sorted_pts = sorted(intersections.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    # Take top 54
    selected_pts = [pt for pt, _ in sorted_pts[:54]]
    
    if len(selected_pts) < 54:
        return -1  # invalid
    
    # Compute actual incidences
    points_arr = np.array(selected_pts, dtype=np.int64)
    lines_arr = np.array(lines_list, dtype=np.int64)
    return _count_incidences(points_arr, lines_arr)


def _greedy_line_selection(candidates, k=27):
    """Greedy submodular selection of k lines from candidates to maximize incidences."""
    selected = []
    remaining = list(candidates)
    
    for _ in range(k):
        best_gain = -1
        best_line = None
        for i, line in enumerate(remaining):
            test_set = selected + [line]
            score = _score_line_set(test_set)
            if score > best_gain:
                best_gain = score
                best_line = line
                best_idx = i
        if best_line is None:
            break
        selected.append(best_line)
        remaining.pop(best_idx)
    
    return selected


def _local_search_lines(lines_list, candidates, deadline, rng):
    """Local search: try swapping one selected line with one candidate."""
    current_lines = list(lines_list)
    current_score = _score_line_set(current_lines)
    best_lines = list(current_lines)
    best_score = current_score
    
    # Build candidate pool: lines not in current set
    current_set = set(tuple(l) for l in current_lines)
    candidates_not_selected = [l for l in candidates if tuple(l) not in current_set]
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline - 30 and T > T_min:
        # Pick a random line to remove
        idx = rng.integers(0, len(current_lines))
        removed = current_lines[idx]
        
        # Pick a random candidate to add
        if not candidates_not_selected:
            break
        cand_idx = rng.integers(0, len(candidates_not_selected))
        added = candidates_not_selected[cand_idx]
        
        new_lines = list(current_lines)
        new_lines[idx] = added
        new_score = _score_line_set(new_lines)
        
        if new_score > best_score:
            current_lines = new_lines
            best_lines = list(new_lines)
            best_score = new_score
            T = max(T * 0.98, T_min)
        else:
            delta = new_score - best_score
            if delta < 0 and rng.random() < np.exp(delta / T):
                current_lines = new_lines
        T *= cooling
    
    return best_lines, best_score


def _generate_candidate_lines(points_grid):
    """Generate candidate lines from a grid of points."""
    n = len(points_grid)
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points_grid[i], points_grid[j])
            if key is not None:
                line_set.add(key)
    return list(line_set)


def _initial_grid_points(m, n):
    """Generate m x n grid points."""
    points = [(x, y) for x in range(1, m + 1) for y in range(1, n + 1)]
    return np.array(points, dtype=np.int64)


def _sheared_grid_points(m, n, shear):
    """Generate sheared grid points."""
    points = []
    for x in range(1, m + 1):
        for y in range(1, n + 1):
            points.append((x, y + shear * x))
    return np.array(points, dtype=np.int64)


def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    safety_margin = 30
    rng = np.random.default_rng(123)
    
    best_points = None
    best_lines = None
    best_count = -1
    
    # Phase 1: Try various grid configurations as starting points
    # For each grid, generate candidate lines, then do line-set optimization
    
    grid_configs = [
        (6, 9, 0), (7, 8, 0), (5, 11, 0), (8, 7, 0),
        (6, 9, 1), (6, 9, -1), (7, 8, 1), (7, 8, -1),
        (5, 11, 1), (5, 11, -1), (8, 7, 1), (8, 7, -1),
        (6, 9, 2), (6, 9, -2), (7, 8, 2), (7, 8, -2),
    ]
    
    for m, n, shear in grid_configs:
        if time.time() > deadline - safety_margin:
            break
        
        if shear == 0:
            pts = _initial_grid_points(m, n)
        else:
            pts = _sheared_grid_points(m, n, shear)
        
        if len(pts) < 54:
            continue
        
        # Take first 54 points if more
        pts = pts[:54]
        
        # Generate candidate lines from these points
        candidates = _generate_candidate_lines(pts)
        
        if len(candidates) < 27:
            continue
        
        # Greedy selection
        selected_lines = _greedy_line_selection(candidates, 27)
        
        if len(selected_lines) < 27:
            continue
        
        score = _score_line_set(selected_lines)
        if score > best_count:
            # Get the actual points
            n_lines = len(selected_lines)
            intersections = {}
            for i in range(n_lines):
                for j in range(i + 1, n_lines):
                    pt = _point_from_two_lines(selected_lines[i], selected_lines[j])
                    if pt is not None:
                        if pt not in intersections:
                            intersections[pt] = set()
                        intersections[pt].add(i)
                        intersections[pt].add(j)
            
            sorted_pts = sorted(intersections.items(), key=lambda kv: len(kv[1]), reverse=True)
            if len(sorted_pts) >= 54:
                sel_pts = [pt for pt, _ in sorted_pts[:54]]
                points_arr = np.array(sel_pts, dtype=np.int64)
                lines_arr = np.array(selected_lines, dtype=np.int64)
                if _is_valid(points_arr, lines_arr):
                    actual_count = _count_incidences(points_arr, lines_arr)
                    if actual_count > best_count:
                        best_count = actual_count
                        best_points = points_arr.copy()
                        best_lines = lines_arr.copy()
    
    # Phase 2: Local search on the best line set found
    if best_lines is not None and time.time() < deadline - safety_margin:
        # Generate a larger candidate pool from the best points
        candidates = _generate_candidate_lines(best_points)
        # Add some random lines
        for _ in range(200):
            A = rng.integers(-3, 4)
            B = rng.integers(-3, 4)
            if A == 0 and B == 0:
                continue
            C = rng.integers(-10, 11)
            key = _normalize_line(int(A), int(B), int(C))
            if key[0] != 0 or key[1] != 0:
                if tuple(key) not in [tuple(l) for l in candidates]:
                    candidates.append(list(key))
        
        # Local search
        new_lines, new_score = _local_search_lines(best_lines, candidates, deadline, rng)
        if new_score > best_count:
            # Get points for new line set
            n_lines = len(new_lines)
            intersections = {}
            for i in range(n_lines):
                for j in range(i + 1, n_lines):
                    pt = _point_from_two_lines(new_lines[i], new_lines[j])
                    if pt is not None:
                        if pt not in intersections:
                            intersections[pt] = set()
                        intersections[pt].add(i)
                        intersections[pt].add(j)
            
            sorted_pts = sorted(intersections.items(), key=lambda kv: len(kv[1]), reverse=True)
            if len(sorted_pts) >= 54:
                sel_pts = [pt for pt, _ in sorted_pts[:54]]
                points_arr = np.array(sel_pts, dtype=np.int64)
                lines_arr = np.array(new_lines, dtype=np.int64)
                if _is_valid(points_arr, lines_arr):
                    actual_count = _count_incidences(points_arr, lines_arr)
                    if actual_count > best_count:
                        best_count = actual_count
                        best_points = points_arr.copy()
                        best_lines = lines_arr.copy()
    
    # Phase 3: Point refinement - try moving individual points to increase incidences
    if best_points is not None and time.time() < deadline - safety_margin:
        points = best_points.copy()
        lines = best_lines.copy()
        count = best_count
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < deadline - safety_margin and T > T_min:
            idx = rng.integers(0, 54)
            dx = rng.integers(-3, 4)
            dy = rng.integers(-3, 4)
            if dx == 0 and dy == 0:
                continue
            
            new_points = points.copy()
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
            
            # Check for duplicates
            valid = True
            pt_set = set()
            for i in range(54):
                x, y = int(new_points[i, 0]), int(new_points[i, 1])
                if (x, y) in pt_set:
                    valid = False
                    break
                pt_set.add((x, y))
            
            if not valid:
                T *= cooling
                continue
            
            # Recompute best lines for new points
            x = new_points[:, 0:1].astype(np.int64)
            y = new_points[:, 1:2].astype(np.int64)
            
            # Find all lines through pairs
            line_counts = {}
            for i in range(54):
                for j in range(i + 1, 54):
                    key = _line_from_two_points(new_points[i], new_points[j])
                    if key is not None:
                        line_counts[key] = line_counts.get(key, 0) + 1
            
            sorted_lines = sorted(line_counts.items(), key=lambda kv: kv[1], reverse=True)
            selected = [key for key, cnt in sorted_lines[:27]]
            
            if len(selected) < 27:
                # Pad
                existing_set = set(selected)
                for i in range(54):
                    if len(selected) >= 27:
                        break
                    px, py = int(new_points[i, 0]), int(new_points[i, 1])
                    for A, B, C in [(1, 0, -px), (0, 1, -py), (1, 1, -(px+py)), (1, -1, py-px)]:
                        key = _normalize_line(A, B, C)
                        if key not in existing_set and not (key[0] == 0 and key[1] == 0):
                            selected.append(key)
                            existing_set.add(key)
                            break
            
            lines_arr = np.array(selected[:27], dtype=np.int64)
            
            if not _is_valid(new_points, lines_arr):
                T *= cooling
                continue
            
            new_count = _count_incidences(new_points, lines_arr)
            
            if new_count > count:
                points = new_points
                lines = lines_arr
                count = new_count
                if count > best_count:
                    best_count = count
                    best_points = points.copy()
                    best_lines = lines.copy()
                T = max(T * 0.98, T_min)
            else:
                delta = new_count - count
                if delta < 0 and rng.random() < np.exp(delta / T):
                    points = new_points
                    lines = lines_arr
            T *= cooling
    
    # Final validation
    if best_points is None or best_lines is None:
        # Fallback: simple grid
        pts = _initial_grid_points(6, 9)[:54]
        x = pts[:, 0:1].astype(np.int64)
        y = pts[:, 1:2].astype(np.int64)
        line_counts = {}
        for i in range(54):
            for j in range(i + 1, 54):
                key = _line_from_two_points(pts[i], pts[j])
                if key is not None:
                    line_counts[key] = line_counts.get(key, 0) + 1
        sorted_lines = sorted(line_counts.items(), key=lambda kv: kv[1], reverse=True)
        selected = [key for key, cnt in sorted_lines[:27]]
        lines_arr = np.array(selected, dtype=np.int64)
        return pts, lines_arr
    
    return best_points, best_lines
# EVOLVE-BLOCK-END
