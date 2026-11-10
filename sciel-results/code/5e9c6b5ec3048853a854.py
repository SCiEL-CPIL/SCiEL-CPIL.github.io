# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so first nonzero of (A,B) is positive, and gcd divides out."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    # Make first nonzero positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    # Divide by gcd
    g = 0
    for v in (A, B, C):
        g = _gcd(g, abs(v))
    if g > 1:
        A //= g
        B //= g
        C //= g
    return (A, B, C)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_incidence(points, line):
    """Count how many points lie on the line A*x + B*y + C = 0."""
    A, B, C = line
    count = 0
    for x, y in points:
        if A * x + B * y + C == 0:
            count += 1
    return count

def _generate_candidate_lines(points, max_slope=10, max_intercept=20):
    """Generate candidate lines from various slopes and intercepts."""
    candidates = set()
    xs = points[:, 0]
    ys = points[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    
    # Vertical lines
    for x in range(min_x, max_x + 1):
        candidates.add((1, 0, -x))
    
    # Horizontal lines
    for y in range(min_y, max_y + 1):
        candidates.add((0, 1, -y))
    
    # Diagonal and other slope lines
    for slope in range(-max_slope, max_slope + 1):
        if slope == 0:
            continue
        # y = slope*x + b  =>  slope*x - y + b = 0
        # b ranges to cover the grid
        for b in range(-max_intercept, max_intercept + 1):
            line = (slope, -1, b)
            norm = _normalize_line(*line)
            if norm:
                candidates.add(norm)
    
    # Also try lines through pairs of points (to catch any missed)
    n = len(points)
    for i in range(min(n, 60)):
        for j in range(i + 1, min(n, 60)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm:
                candidates.add(norm)
    
    return list(candidates)

def _greedy_select_lines(points, num_lines=27):
    """Greedy: repeatedly pick the line with most new incidences."""
    candidates = _generate_candidate_lines(points)
    
    # Precompute incidence for each candidate
    line_incidences = {}
    for line in candidates:
        inc = _count_incidence(points, line)
        if inc >= 2:  # Only consider lines hitting at least 2 points
            line_incidences[line] = inc
    
    # Sort by incidence descending
    sorted_lines = sorted(line_incidences.keys(), key=lambda l: line_incidences[l], reverse=True)
    
    selected = []
    selected_set = set()
    # Track which points are already "covered" (on at least one selected line)
    # Actually we want total incidences, so we just sum up incidences of selected lines
    # But we need to avoid duplicate lines
    
    # Greedy: pick line with highest incidence not already selected
    # Since we want to maximize sum of incidences, and lines can share points,
    # we just pick the top num_lines by incidence (no conflict since we count all incidences)
    
    # But we must avoid duplicate lines (same geometric line)
    # The normalization handles that
    
    selected = sorted_lines[:num_lines]
    return selected

def _evaluate(points, lines):
    """Count total incidences."""
    total = 0
    for line in lines:
        A, B, C = line
        for x, y in points:
            if A * x + B * y + C == 0:
                total += 1
    return total

def _try_grid(w, h):
    """Try a w x h grid and find best 27 lines."""
    points = np.array([[x, y] for x in range(w) for y in range(h)], dtype=np.int64)
    if len(points) != 54:
        return None, 0
    
    lines = _greedy_select_lines(points, 27)
    if len(lines) < 27:
        return None, 0
    
    score = _evaluate(points, lines)
    return points, lines, score

def _try_rect_grid(w, h):
    """Try w x h grid."""
    points = np.array([[x, y] for x in range(w) for y in range(h)], dtype=np.int64)
    if len(points) != 54:
        return None, None, 0
    
    # Generate all possible lines
    candidates = set()
    # Vertical
    for x in range(w):
        candidates.add((1, 0, -x))
    # Horizontal
    for y in range(h):
        candidates.add((0, 1, -y))
    # All lines through pairs
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm:
                candidates.add(norm)
    
    # Count incidences
    line_incs = {}
    for line in candidates:
        A, B, C = line
        inc = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                inc += 1
        if inc >= 2:
            line_incs[line] = inc
    
    # Sort and pick top 27
    sorted_lines = sorted(line_incs.keys(), key=lambda l: line_incs[l], reverse=True)
    selected = sorted_lines[:27]
    
    if len(selected) < 27:
        return None, None, 0
    
    score = sum(line_incs[l] for l in selected)
    return points, selected, score

def _try_parametric_grid(w, h, offset_x=0, offset_y=0):
    """Try a shifted grid."""
    points = np.array([[x + offset_x, y + offset_y] for x in range(w) for y in range(h)], dtype=np.int64)
    if len(points) != 54:
        return None, None, 0
    
    candidates = set()
    for x in range(w + offset_x, w + offset_x):
        pass
    for x in set(points[:, 0]):
        candidates.add((1, 0, -x))
    for y in set(points[:, 1]):
        candidates.add((0, 1, -y))
    
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm:
                candidates.add(norm)
    
    line_incs = {}
    for line in candidates:
        A, B, C = line
        inc = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                inc += 1
        if inc >= 2:
            line_incs[line] = inc
    
    sorted_lines = sorted(line_incs.keys(), key=lambda l: line_incs[l], reverse=True)
    selected = sorted_lines[:27]
    
    if len(selected) < 27:
        return None, None, 0
    
    score = sum(line_incs[l] for l in selected)
    return points, selected, score

def _try_hexagonal():
    """Try a hexagonal/lattice structure."""
    # 54 points in a hexagonal arrangement
    # Center at (0,0), rings
    points = set()
    # Ring 0: (0,0)
    points.add((0, 0))
    # Ring 1: 6 points
    for a in range(6):
        x = [1, 0, -1, -1, 0, 1][a]
        y = [0, 1, 1, 0, -1, -1][a]
        points.add((x, y))
    # Ring 2: 12 points
    for a in range(12):
        # Hexagonal ring 2
        if a == 0: points.add((2, 0))
        elif a == 1: points.add((1, 2))
        elif a == 2: points.add((-1, 2))
        elif a == 3: points.add((-2, 1))
        elif a == 4: points.add((-2, -1))
        elif a == 5: points.add((-1, -2))
        elif a == 6: points.add((1, -2))
        elif a == 7: points.add((2, -1))
        elif a == 8: points.add((1, 1))
        elif a == 9: points.add((0, 2))
        elif a == 10: points.add((-1, 1))
        elif a == 11: points.add((-1, 0))
    # We have 1 + 6 + 12 = 19 points. Need 54.
    # Let's use a different approach: just use grids with different aspect ratios
    
    return None, None, 0

def _try_staggered_grid(w, h):
    """Staggered grid: odd rows shifted by 0.5 (but we need integers).
    Use a brick-wall pattern."""
    points = []
    for row in range(h):
        for col in range(w):
            x = col
            y = row
            if row % 2 == 1:
                x = col + 0.5  # not integer
            points.append((x, y))
    # This doesn't work for integer lattice. Skip.
    return None, None, 0

def _brute_force_small_grid(w, h):
    """For small grids, try all line subsets via greedy + local search."""
    points = np.array([[x, y] for x in range(w) for y in range(h)], dtype=np.int64)
    if len(points) != 54:
        return None, None, 0
    
    # Generate all lines through pairs of points
    n = len(points)
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm:
                line_set.add(norm)
    
    # Count incidences
    line_incs = {}
    for line in line_set:
        A, B, C = line
        inc = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                inc += 1
        line_incs[line] = inc
    
    # Sort by incidence
    sorted_lines = sorted(line_incs.keys(), key=lambda l: line_incs[l], reverse=True)
    
    # Greedy select top 27
    selected = sorted_lines[:27]
    if len(selected) < 27:
        return None, None, 0
    
    score = sum(line_incs[l] for l in selected)
    return points, selected, score

def _local_search(points, lines, line_incs, iterations=100):
    """Try swapping lines to improve score."""
    selected = list(lines)
    selected_set = set(selected)
    current_score = sum(line_incs[l] for l in selected)
    
    # All candidate lines not in selected
    all_lines = list(line_incs.keys())
    unselected = [l for l in all_lines if l not in selected_set]
    
    improved = True
    while improved and iterations > 0:
        improved = False
        iterations -= 1
        for i in range(len(selected)):
            for j in range(len(unselected)):
                new_score = current_score - line_incs[selected[i]] + line_incs[unselected[j]]
                if new_score > current_score:
                    # Make the swap
                    selected[i] = unselected[j]
                    selected_set.discard(unselected[j])
                    selected_set.add(unselected[j])  # wait, this is wrong
                    # Actually:
                    old_line = selected[i]
                    new_line = unselected[j]
                    selected[i] = new_line
                    selected_set.remove(old_line)
                    selected_set.add(new_line)
                    unselected.remove(new_line)
                    unselected.append(old_line)
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return selected, current_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 580.0  # 580 seconds to be safe
    
    best_score = 0
    best_points = None
    best_lines = None
    
    # Try various grid factorizations of 54
    factorizations = [
        (1, 54), (2, 27), (3, 18), (6, 9),
        (54, 1), (27, 2), (18, 3), (9, 6),
    ]
    
    # Also try non-rectangular arrangements
    arrangements = [
        ('grid', 6, 9),
        ('grid', 9, 6),
        ('grid', 3, 18),
        ('grid', 18, 3),
        ('grid', 2, 27),
        ('grid', 27, 2),
    ]
    
    for arr_type, w, h in arrangements:
        if time.time() > deadline:
            break
        
        if arr_type == 'grid':
            points, lines, score = _brute_force_small_grid(w, h)
            if points is not None and score > best_score:
                best_score = score
                best_points = points
                best_lines = lines
    
    # Try shifted grids
    for w, h in [(6, 9), (9, 6)]:
        if time.time() > deadline:
            break
        for ox in range(-2, 3):
            for oy in range(-2, 3):
                if time.time() > deadline:
                    break
                points, lines, score = _try_parametric_grid(w, h, ox, oy)
                if points is not None and score > best_score:
                    best_score = score
                    best_points = points
                    best_lines = lines
    
    # Try a more sophisticated approach: build points to maximize collinearities
    # Use a 6x9 grid but with local search on line selection
    if time.time() < deadline:
        w, h = 6, 9
        points = np.array([[x, y] for x in range(w) for y in range(h)], dtype=np.int64)
        n = len(points)
        
        line_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                norm = _normalize_line(A, B, C)
                if norm:
                    line_set.add(norm)
        
        line_incs = {}
        for line in line_set:
            A, B, C = line
            inc = 0
            for x, y in points:
                if A * x + B * y + C == 0:
                    inc += 1
            line_incs[line] = inc
        
        sorted_lines = sorted(line_incs.keys(), key=lambda l: line_incs[l], reverse=True)
        selected = sorted_lines[:27]
        
        if len(selected) >= 27:
            selected, score = _local_search(points, selected, line_incs, iterations=50)
            if score > best_score:
                best_score = score
                best_points = points
                best_lines = selected
    
    # Fallback: use the Szemeredi-Trotter construction
    if best_points is None or best_lines is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    # Ensure correct format
    best_points = np.array(best_points, dtype=np.int64)
    best_lines = np.array(best_lines, dtype=np.int64)
    
    # Validate: no duplicate points
    unique_points = np.unique(best_points, axis=0)
    if len(unique_points) != 54:
        # Fallback
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    # Validate: no duplicate lines, no zero lines
    valid_lines = []
    seen = set()
    for line in best_lines:
        A, B, C = int(line[0]), int(line[1]), int(line[2])
        if A == 0 and B == 0:
            continue
        norm = _normalize_line(A, B, C)
        if norm is None:
            continue
        if norm not in seen:
            seen.add(norm)
            valid_lines.append(norm)
    
    if len(valid_lines) >= 27:
        best_lines = np.array(valid_lines[:27], dtype=np.int64)
    else:
        # Fallback
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
