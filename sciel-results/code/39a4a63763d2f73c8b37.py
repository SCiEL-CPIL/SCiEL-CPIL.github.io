# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time
from itertools import combinations
import random


def _line_key(A, B, C):
    """Normalize a line (A,B,C) for deduplication."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidence(points, lines):
    """Count point-line incidences exactly."""
    count = 0
    for i in range(points.shape[0]):
        x, y = points[i]
        for j in range(lines.shape[0]):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _incidence_of_point(x, y, lines):
    """Count how many lines pass through point (x,y)."""
    inc = 0
    for j in range(lines.shape[0]):
        A, B, C = lines[j]
        if A * x + B * y + C == 0:
            inc += 1
    return inc


def _incidence_of_line(A, B, C, points):
    """Count how many points lie on line Ax+By+C=0."""
    inc = 0
    for i in range(points.shape[0]):
        x, y = points[i]
        if A * x + B * y + C == 0:
            inc += 1
    return inc


def _generate_lines_from_points(points, n_lines):
    """Generate candidate lines from all pairs of points, then select best n_lines."""
    n = points.shape[0]
    candidates = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key not in candidates:
                candidates[key] = (A, B, C)
    
    # Score each candidate line by number of incidences
    scored = []
    for key, (A, B, C) in candidates.items():
        inc = 0
        for i in range(n):
            x, y = points[i]
            if A * x + B * y + C == 0:
                inc += 1
        scored.append((inc, key, A, B, C))
    
    # Sort by incidence count descending
    scored.sort(key=lambda x: -x[0])
    
    # Select top n_lines unique lines
    selected = []
    seen = set()
    for inc, key, A, B, C in scored:
        if key not in seen and len(selected) < n_lines:
            selected.append((A, B, C))
            seen.add(key)
    
    # If we don't have enough, add arbitrary lines
    if len(selected) < n_lines:
        for c in range(0, 100):
            if len(selected) >= n_lines:
                break
            line = (0, 1, -c)
            key = _line_key(line[0], line[1], line[2])
            if key not in seen:
                selected.append(line)
                seen.add(key)
    
    return np.array(selected[:n_lines], dtype=np.int64)


def _local_search_points(points, lines, deadline, max_iter=50):
    """Local search to improve point positions."""
    best_incidence = _count_incidence(points, lines)
    n = points.shape[0]
    m = lines.shape[0]
    
    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        improved = False
        
        for i in range(n):
            old_x, old_y = int(points[i, 0]), int(points[i, 1])
            best_new = (old_x, old_y)
            best_new_inc = best_incidence
            
            old_inc = _incidence_of_point(old_x, old_y, lines)
            
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    # Check for duplicate points
                    dup = False
                    for k in range(n):
                        if k != i and points[k, 0] == new_x and points[k, 1] == new_y:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    new_inc = _incidence_of_point(new_x, new_y, lines)
                    total_inc = best_incidence - old_inc + new_inc
                    if total_inc > best_new_inc:
                        best_new_inc = total_inc
                        best_new = (new_x, new_y)
            
            if best_new != (old_x, old_y):
                points[i] = best_new
                best_incidence = best_new_inc
                improved = True
        
        if not improved:
            break
    
    return points, best_incidence


def _local_search_lines(points, lines, deadline, max_iter=50):
    """Local search to improve line positions."""
    best_incidence = _count_incidence(points, lines)
    n = points.shape[0]
    m = lines.shape[0]
    
    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        improved = False
        
        for j in range(m):
            A, B, C = int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])
            old_inc = _incidence_of_line(A, B, C, points)
            
            best_new_line = (A, B, C)
            best_new_inc = best_incidence
            
            # Try nearby lines
            for dA in range(-3, 4):
                for dB in range(-3, 4):
                    for dC in range(-10, 11):
                        if dA == 0 and dB == 0 and dC == 0:
                            continue
                        new_A, new_B, new_C = A + dA, B + dB, C + dC
                        if new_A == 0 and new_B == 0:
                            continue
                        
                        # Check for duplicate lines
                        norm = _line_key(new_A, new_B, new_C)
                        is_dup = False
                        for k in range(m):
                            if k == j:
                                continue
                            if _line_key(int(lines[k, 0]), int(lines[k, 1]), int(lines[k, 2])) == norm:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        
                        new_inc = _incidence_of_line(new_A, new_B, new_C, points)
                        total_inc = best_incidence - old_inc + new_inc
                        if total_inc > best_new_inc:
                            best_new_inc = total_inc
                            best_new_line = (new_A, new_B, new_C)
            
            if best_new_line != (A, B, C):
                lines[j] = best_new_line
                best_incidence = best_new_inc
                improved = True
        
        if not improved:
            break
    
    return lines, best_incidence


def _refine_configuration(points, lines, deadline):
    """Iteratively refine both points and lines."""
    best_incidence = _count_incidence(points, lines)
    
    for round_num in range(10):
        if time.time() > deadline - 30:
            break
        
        points, inc1 = _local_search_points(points, lines, deadline, max_iter=20)
        lines, inc2 = _local_search_lines(points, lines, deadline, max_iter=20)
        
        if inc2 <= best_incidence:
            break
        best_incidence = inc2
    
    return points, lines, best_incidence


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses clique-star hybrid: 27 points on a vertical line + 27 on a 3x9 grid,
    with one line for the vertical clique and 26 lines optimized for the grid.
    """
    deadline = time.time() + 570  # safety margin
    best_points = None
    best_lines = None
    best_incidence = 0
    
    # Strategy 1: Clique-star hybrid
    # 27 points on vertical line x=0, y=0..26
    # 27 points on 3x9 grid: x=1,2,3; y=0..8
    grid_x = [1, 2, 3]
    grid_y = list(range(9))
    
    clique_points = [(0, y) for y in range(27)]
    grid_points = [(x, y) for x in grid_x for y in grid_y]
    
    all_points = clique_points + grid_points
    points = np.array(all_points, dtype=np.int64)
    
    # Line 0: the vertical line x=0 (passes through all 27 clique points)
    # Lines 1-26: optimize for the grid
    grid_points_arr = np.array(grid_points, dtype=np.int64)
    
    # Generate best 26 lines for the grid
    grid_lines = _generate_lines_from_points(grid_points_arr, 26)
    
    # Combine: vertical line + grid lines
    lines = np.vstack([np.array([[0, 1, 0]], dtype=np.int64), grid_lines])
    
    # Check for duplicate lines
    line_keys = set()
    unique_lines = []
    for i in range(lines.shape[0]):
        key = _line_key(int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2]))
        if key not in line_keys:
            unique_lines.append(lines[i])
            line_keys.add(key)
    
    # If we have duplicates, replace with new lines
    if len(unique_lines) < 27:
        # Add more lines
        for c in range(-50, 50):
            if len(unique_lines) >= 27:
                break
            line = np.array([0, 1, c], dtype=np.int64)
            key = _line_key(0, 1, c)
            if key not in line_keys:
                unique_lines.append(line)
                line_keys.add(key)
    
    lines = np.array(unique_lines[:27], dtype=np.int64)
    
    incidence = _count_incidence(points, lines)
    if incidence > best_incidence:
        best_incidence = incidence
        best_points = points.copy()
        best_lines = lines.copy()
    
    # Refine the configuration
    if time.time() < deadline - 60:
        points, lines, inc = _refine_configuration(points, lines, deadline)
        if inc > best_incidence:
            best_incidence = inc
            best_points = points.copy()
            best_lines = lines.copy()
    
    # Strategy 2: Pure grid with more lines
    if time.time() < deadline - 120:
        # 54 points on a 6x9 grid
        xs = list(range(6))
        ys = list(range(9))
        grid_pts = [(x, y) for x in xs for y in ys]
        pts = np.array(grid_pts, dtype=np.int64)
        
        grid_lines = _generate_lines_from_points(pts, 27)
        
        # Check for duplicates
        line_keys = set()
        unique_lines = []
        for i in range(grid_lines.shape[0]):
            key = _line_key(int(grid_lines[i, 0]), int(grid_lines[i, 1]), int(grid_lines[i, 2]))
            if key not in line_keys:
                unique_lines.append(grid_lines[i])
                line_keys.add(key)
        
        if len(unique_lines) < 27:
            for c in range(-50, 50):
                if len(unique_lines) >= 27:
                    break
                line = np.array([0, 1, c], dtype=np.int64)
                key = _line_key(0, 1, c)
                if key not in line_keys:
                    unique_lines.append(line)
                    line_keys.add(key)
        
        lines = np.array(unique_lines[:27], dtype=np.int64)
        incidence = _count_incidence(pts, lines)
        
        if incidence > best_incidence:
            best_incidence = incidence
            best_points = pts.copy()
            best_lines = lines.copy()
        
        if time.time() < deadline - 60:
            pts, lines, inc = _refine_configuration(pts, lines, deadline)
            if inc > best_incidence:
                best_incidence = inc
                best_points = pts.copy()
                best_lines = lines.copy()
    
    # Strategy 3: Clique-star with different grid dimensions
    if time.time() < deadline - 120:
        # Try 27 on vertical + 27 on 1x27 (horizontal line)
        clique_points = [(0, y) for y in range(27)]
        h_points = [(x, 0) for x in range(1, 28)]
        
        all_pts = clique_points + h_points
        pts = np.array(all_pts, dtype=np.int64)
        
        # Vertical line x=0 gives 27 incidences
        # Horizontal line y=0 gives 27 incidences (but x=0,y=0 is shared)
        # Actually the intersection point (0,0) is in both, so we have 53 unique points
        # Wait, we need 54 points. Let me reconsider.
        
        # Better: 27 on x=0 (y=0..26), 27 on y=0 (x=1..27)
        # Point (0,0) is in clique but not in horizontal (x starts at 1)
        # So 54 unique points
        pts = np.array(clique_points + h_points, dtype=np.int64)
        
        # Line 0: x=0 (27 incidences)
        # Line 1: y=0 (27 incidences, but (0,0) is on both)
        # Remaining 25 lines: optimize
        remaining_points = pts  # all points
        grid_lines = _generate_lines_from_points(remaining_points, 25)
        
        lines_list = [np.array([0, 1, 0], dtype=np.int64), np.array([1, 0, 0], dtype=np.int64)]
        for i in range(grid_lines.shape[0]):
            lines_list.append(grid_lines[i])
        
        lines = np.array(lines_list[:27], dtype=np.int64)
        
        # Check for duplicates
        line_keys = set()
        unique_lines = []
        for i in range(lines.shape[0]):
            key = _line_key(int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2]))
            if key not in line_keys:
                unique_lines.append(lines[i])
                line_keys.add(key)
        
        if len(unique_lines) < 27:
            for c in range(-50, 50):
                if len(unique_lines) >= 27:
                    break
                line = np.array([0, 1, c], dtype=np.int64)
                key = _line_key(0, 1, c)
                if key not in line_keys:
                    unique_lines.append(line)
                    line_keys.add(key)
        
        lines = np.array(unique_lines[:27], dtype=np.int64)
        incidence = _count_incidence(pts, lines)
        
        if incidence > best_incidence:
            best_incidence = incidence
            best_points = pts.copy()
            best_lines = lines.copy()
        
        if time.time() < deadline - 60:
            pts, lines, inc = _refine_configuration(pts, lines, deadline)
            if inc > best_incidence:
                best_incidence = inc
                best_points = pts.copy()
                best_lines = lines.copy()
    
    # Final refinement on the best configuration
    if best_points is not None and time.time() < deadline - 30:
        best_points, best_lines, final_inc = _refine_configuration(
            best_points.copy(), best_lines.copy(), deadline
        )
        if final_inc > best_incidence:
            best_incidence = final_inc
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
