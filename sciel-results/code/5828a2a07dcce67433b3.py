# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients so first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact incidences between points and lines using int64 arithmetic."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _generate_candidate_lines(points):
    """Generate candidate lines from pairs of points, normalized."""
    n = len(points)
    candidate_set = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm is not None:
                candidate_set[norm] = norm
    return list(candidate_set.values())

def _compute_line_weights(points, lines):
    """Compute incidence count for each line."""
    pts = np.asarray(points, dtype=np.int64)
    weights = np.zeros(len(lines), dtype=np.int64)
    for i in range(len(lines)):
        A, B, C = lines[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        weights[i] = np.sum(vals == 0)
    return weights

def _find_best_lines_for_points(points, num_lines):
    """
    Find the best num_lines lines that maximize incidences with given points.
    This is done once to set up a good line set.
    """
    n = len(points)
    # Collect all lines determined by pairs of points
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _normalize_line(points[j][1] - points[i][1], points[i][0] - points[j][0], points[j][0]*points[i][1] - points[i][0]*points[j][1])
            if line is not None:
                line_counts[line] = line_counts.get(line, 0) + 1
    
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Greedily pick top lines
    selected = []
    selected_set = set()
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        if line not in selected_set:
            selected.append(line)
            selected_set.add(line)
            
    # If we don't have enough, add arbitrary distinct lines
    if len(selected) < num_lines:
        # Add lines with no points or low points to fill up
        # We need to ensure they are distinct from selected
        # Simple fallback: add lines with large C values
        existing_norms = set(selected)
        counter = 100
        while len(selected) < num_lines:
            # Try a generic line
            test_line = _normalize_line(1, 1, -counter)
            if test_line not in existing_norms:
                selected.append(test_line)
                existing_norms.add(test_line)
            counter += 1
            
    return np.array(selected, dtype=np.int64)

def _point_incidence_count(point, lines):
    """Count how many lines pass through a single point."""
    x, y = int(point[0]), int(point[1])
    count = 0
    for A, B, C in lines:
        if A * x + B * y + C == 0:
            count += 1
    return count

def _optimize_points_fixed_lines(points, lines, deadline):
    """
    Optimize point positions to maximize incidences with a fixed set of lines.
    Uses local search: move a point to a nearby integer coordinate if it increases incidence.
    """
    pts = points.copy()
    n = len(pts)
    m = len(lines)
    
    # Precompute line coefficients for speed
    lns = lines.astype(np.int64)
    
    # Initial incidence count per point
    point_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        point_counts[i] = _point_incidence_count(pts[i], lns)
    
    total_inc = np.sum(point_counts)
    
    start_time = time.time()
    improved = True
    max_radius = 3 # Search radius for moves
    
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        # Shuffle order to avoid bias? Or just iterate. 
        # Iterating is fine if we allow multiple passes.
        for i in range(n):
            if (time.time() - start_time) > deadline - 10:
                break
            
            old_x, old_y = int(pts[i, 0]), int(pts[i, 1])
            old_count = point_counts[i]
            
            # Try nearby positions
            best_new_x, best_new_y = old_x, old_y
            best_new_count = old_count
            
            # Check neighbors in a box of radius max_radius
            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    # Check validity: integer (yes, by construction)
                    # Check uniqueness: no other point at (new_x, new_y)
                    # Efficient check: 
                    # We can maintain a set of occupied points for O(1) lookup
                    # But for now, let's just check against the array if n is small (54 is small)
                    # Actually, maintaining a set is better.
                    pass # Placeholder for set logic below
                    
            # Let's implement the set-based check properly
            # We need to update the set when a point moves
            # This requires passing the set or returning it.
            # Let's refactor slightly to keep a set of occupied points.
            
            # Re-doing the inner loop with set check
            # We need the set of occupied points. 
            # Let's assume we have a set 'occupied' that is kept up to date.
            # Since we are modifying pts, we need to update occupied.
            
            # To keep it simple and fast, we'll check uniqueness by scanning the small array.
            # 54 points * (2*3+1)^2 candidates * 27 lines is manageable?
            # 54 * 49 * 27 ~ 70k ops per pass. 100 passes ~ 7M ops. Fast enough.
            
            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    # Check if (new_x, new_y) is occupied by another point
                    is_occupied = False
                    for k in range(n):
                        if k != i and int(pts[k, 0]) == new_x and int(pts[k, 1]) == new_y:
                            is_occupied = True
                            break
                    if is_occupied:
                        continue
                    
                    # Calculate new incidence count for this point
                    new_count = 0
                    for j in range(m):
                        A, B, C = lns[j]
                        if A * new_x + B * new_y + C == 0:
                            new_count += 1
                    
                    if new_count > best_new_count:
                        best_new_count = new_count
                        best_new_x, best_new_y = new_x, new_y
            
            # If we found a better spot, move the point
            if best_new_count > old_count:
                pts[i, 0] = best_new_x
                pts[i, 1] = best_new_y
                point_counts[i] = best_new_count
                total_inc += best_new_count - old_count
                improved = True
                
    return pts, total_inc

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Strategy:
    1. Start with 6x9 grid.
    2. Find best 27 lines for this grid.
    3. Optimize point positions to maximize incidences with these fixed lines.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # 1. Generate initial Points (6x9 grid)
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Find best lines for this initial configuration
    lines = _find_best_lines_for_points(points, 27)
    
    # 3. Optimize points with fixed lines
    # We can run this for the remaining time
    points, total_inc = _optimize_points_fixed_lines(points, lines, deadline)
    
    # 4. Final verification and cleanup
    # Check for duplicate points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != len(points):
        # Fallback to grid if something went wrong
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = _find_best_lines_for_points(points, 27)
        
    # Ensure lines are valid (no A=B=0)
    valid_mask = ~((lines[:, 0] == 0) & (lines[:, 1] == 0))
    if not np.all(valid_mask):
        # Fallback
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = _find_best_lines_for_points(points, 27)
        
    # Check for duplicate lines
    normalized_lines = [_normalize_line(int(l[0]), int(l[1]), int(l[2])) for l in lines]
    seen = set()
    unique_norms = []
    for nl in normalized_lines:
        if nl is not None and nl not in seen:
            seen.add(nl)
            unique_norms.append(nl)
            
    if len(unique_norms) != len(lines):
        # If duplicates exist, we need to replace some with distinct lines
        # This is a safety net, should not happen if _find_best_lines_for_points is correct
        # But _optimize_points_fixed_lines doesn't change lines.
        # So if initial lines were unique, they remain unique.
        # However, if we fell back, we regenerate.
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = _find_best_lines_for_points(points, 27)
        
    return points, lines

# EVOLVE-BLOCK-END
