# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import product, combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Count total incidences between points and lines.
    points: (N, 2) array
    lines: (K, 3) array
    Returns: total incidences (int)
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _find_integer_solutions(A_mat, b_vec, max_coord=20, max_attempts=100):
    """
    Find integer solutions to A_mat * x = b_vec.
    A_mat is (m, 2), b_vec is (m,).
    Returns a list of (x, y) integer tuples.
    """
    if len(A_mat) == 0:
        return []
    
    # If we have at least 2 equations, we can solve for x, y directly
    # A1 x + B1 y = C1
    # A2 x + B2 y = C2
    # Determinant D = A1 B2 - A2 B1
    # If D != 0, unique solution. Check if integer.
    # If D == 0, either no solution or infinite (line).
    
    # Try all pairs of equations to find a candidate
    # If consistent, the solution is unique (if D != 0) or on a line (if D == 0).
    
    # Let's pick the first two independent equations.
    # If they are dependent, try others.
    
    for i in range(len(A_mat)):
        for j in range(i+1, len(A_mat)):
            A1, B1 = A_mat[i]
            A2, B2 = A_mat[j]
            C1, C2 = b_vec[i], b_vec[j]
            
            D = A1 * B2 - A2 * B1
            if D == 0:
                continue
            
            # x = (C1 B2 - C2 B1) / D
            # y = (A1 C2 - A2 C1) / D
            
            num_x = C1 * B2 - C2 * B1
            num_y = A1 * C2 - A2 * C1
            
            if num_x % D != 0 or num_y % D != 0:
                continue
            
            x = num_x // D
            y = num_y // D
            
            # Check if this solution satisfies all other equations
            valid = True
            for k in range(len(A_mat)):
                if A_mat[k][0] * x + A_mat[k][1] * y != b_vec[k]:
                    valid = False
                    break
            
            if valid and abs(x) <= max_coord and abs(y) <= max_coord:
                return [(x, y)]
                
    return []

def _generate_cluster_config(num_points=54, num_lines=27, max_coord=10):
    """
    Generate a configuration by defining clusters of points on lines.
    Strategy:
    1. Define a set of "structural" lines (e.g., grid lines, diagonals).
    2. Assign points to these lines such that intersections are utilized.
    3. Ensure all points are distinct and within bounds.
    """
    points = set()
    lines = []
    
    # We want high incidence. 
    # A grid of size m x n has m vertical, n horizontal lines.
    # Incidences = 2 * m * n.
    # For 54 points, 6x9 grid: 15 lines. Incidence = 108.
    # We have 27 lines. We can add diagonals.
    
    # Let's try to construct a dense configuration.
    # Idea: Use a subset of the 54 points to form a grid, and place remaining points at intersections of additional lines.
    
    # Step 1: Choose a base grid.
    # Let's iterate over possible grid dimensions (w, h) such that w*h <= 54.
    best_score = -1
    best_points = None
    best_lines = None
    
    # Possible grid sizes
    for w in range(1, 12):
        for h in range(1, 12):
            if w * h > 54:
                continue
                
            # Define grid points
            # Center the grid around 0 to keep coords small
            offset_x = -w // 2
            offset_y = -h // 2
            
            grid_points = [(i + offset_x, j + offset_y) for i in range(w) for j in range(h)]
            
            # Base lines: vertical and horizontal
            base_lines = []
            for i in range(w):
                base_lines.append((1, 0, -(i + offset_x)))
            for j in range(h):
                base_lines.append((0, 1, -(j + offset_y)))
            
            # Current points set
            current_points = set(grid_points)
            
            # We have 27 - len(base_lines) slots for extra lines.
            # And 54 - len(grid_points) slots for extra points.
            
            remaining_lines = 27 - len(base_lines)
            remaining_points = 54 - len(grid_points)
            
            if remaining_lines < 0:
                continue
                
            # Try to add diagonal lines or other lines that pass through existing points.
            # To maximize incidence, new lines should pass through as many existing points as possible.
            
            # Candidate lines for the grid:
            # Diagonals with slope 1, -1, etc.
            candidate_lines = []
            
            # Slopes to consider
            slopes = [(1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (-1, 2), (3, 1), (1, 3)]
            
            for A, B in slopes:
                # Calculate C for all grid points
                c_vals = set()
                for x, y in grid_points:
                    c = -(A * x + B * y)
                    c_vals.add(c)
                
                # Sort C values by how many points they cover
                # Count points for each C
                c_counts = {}
                for x, y in grid_points:
                    c = -(A * x + B * y)
                    c_counts[c] = c_counts.get(c, 0) + 1
                    
                # Get top C values
                sorted_cs = sorted(c_counts.items(), key=lambda x: x[1], reverse=True)
                
                for c, count in sorted_cs:
                    if count >= 2: # Only add lines with at least 2 points
                        candidate_lines.append((A, B, c, count))
                        
            # Sort candidates by count descending
            candidate_lines.sort(key=lambda x: x[3], reverse=True)
            
            # Select top candidates up to remaining_lines
            selected_extra_lines = []
            used_c = set()
            for A, B, c, count in candidate_lines:
                key = _normalize_line(A, B, c)
                if key in [tuple(l) for l in selected_extra_lines]:
                    continue
                if key in [tuple(l) for l in base_lines]:
                    continue
                selected_extra_lines.append(key)
                if len(selected_extra_lines) >= remaining_lines:
                    break
            
            all_lines = base_lines + selected_extra_lines
            
            # If we don't have enough lines, add random lines or lines through single points?
            # Adding lines through single points adds 1 incidence.
            # Adding a line through 2 points adds 2 incidences.
            # It's always better to find lines with more points.
            # If we still have slots, we can just pick any valid line not already present.
            # But wait, we must return exactly 27 lines.
            
            # Now, do we have enough points?
            # We have len(grid_points) points. We need 54.
            # If len(grid_points) < 54, we need to add points.
            # Where to add them?
            # To maximize incidence, new points should lie on as many of the 27 lines as possible.
            # The "best" spots are intersections of lines.
            
            # Find all intersection points of the selected lines.
            intersections = set()
            for i in range(len(all_lines)):
                for j in range(i+1, len(all_lines)):
                    A1, B1, C1 = all_lines[i]
                    A2, B2, C2 = all_lines[j]
                    D = A1 * B2 - A2 * B1
                    if D == 0:
                        continue
                    num_x = C1 * B2 - C2 * B1
                    num_y = A1 * C2 - A2 * C1
                    if num_x % D != 0 or num_y % D != 0:
                        continue
                    x = num_x // D
                    y = num_y // D
                    if abs(x) <= max_coord and abs(y) <= max_coord:
                        intersections.add((x, y))
                        
            # Prioritize intersections that are not already in grid_points
            new_points_candidates = [p for p in intersections if p not in current_points]
            
            # Sort new points by number of lines they lie on
            # Precompute line incidence for each candidate
            def count_lines_through(p):
                x, y = p
                cnt = 0
                for A, B, C in all_lines:
                    if A * x + B * y + C == 0:
                        cnt += 1
                return cnt
                
            new_points_candidates.sort(key=count_lines_through, reverse=True)
            
            # Add top candidates until we have 54 points
            for p in new_points_candidates:
                if len(current_points) >= 54:
                    break
                current_points.add(p)
                
            # If we still don't have 54 points, we must add random points.
            # This will lower the score, but we need valid output.
            if len(current_points) < 54:
                # Generate random points in a larger box
                while len(current_points) < 54:
                    x = random.randint(-max_coord, max_coord)
                    y = random.randint(-max_coord, max_coord)
                    if (x, y) not in current_points:
                        current_points.add((x, y))
                        
            # If we have more than 54 points (shouldn't happen if we break), trim
            # Actually, we added exactly up to 54.
            
            # Ensure we have exactly 27 lines.
            # If we have fewer, add random lines.
            while len(all_lines) < 27:
                A = random.randint(-5, 5)
                B = random.randint(-5, 5)
                C = random.randint(-20, 20)
                if A == 0 and B == 0:
                    continue
                key = _normalize_line(A, B, C)
                if key not in all_lines:
                    all_lines.append(key)
                    
            # Convert to arrays
            pts_list = list(current_points)
            pts_arr = np.array(pts_list, dtype=np.int64)
            lines_arr = np.array(all_lines, dtype=np.int64)
            
            score = _count_incidences(pts_arr, lines_arr)
            
            if score > best_score:
                best_score = score
                best_points = pts_arr
                best_lines = lines_arr
                
    return best_points, best_lines, best_score

def _local_search_improve(points, lines, max_time=10):
    """
    Simple local search to improve the configuration.
    Try moving individual points to nearby integer coordinates.
    """
    start_time = time.time()
    deadline = start_time + max_time
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = _count_incidences(current_points, current_lines)
    
    used_points = set([tuple(p) for p in current_points])
    
    while time.time() < deadline - 1:
        # Pick a random point
        idx = random.randint(0, len(current_points) - 1)
        old_pt = tuple(current_points[idx])
        
        # Try nearby moves
        improved = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt in used_points:
                    continue
                
                # Calculate delta
                # Old contribution
                old_contrib = 0
                for A, B, C in current_lines:
                    if A * old_pt[0] + B * old_pt[1] + C == 0:
                        old_contrib += 1
                        
                # New contribution
                new_contrib = 0
                for A, B, C in current_lines:
                    if A * new_pt[0] + B * new_pt[1] + C == 0:
                        new_contrib += 1
                        
                if new_contrib > old_contrib:
                    # Accept
                    used_points.remove(old_pt)
                    used_points.add(new_pt)
                    current_points[idx] = new_pt
                    current_score += (new_contrib - old_contrib)
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            # No improvement in local neighborhood, maybe try a random jump?
            # Or just continue.
            # To avoid getting stuck, we could do a random restart or larger jump.
            # For now, just continue to next point.
            pass
            
    return current_points, current_lines, current_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses combinatorial enumeration of grid-based structures followed by local search.
    """
    start_time = time.time()
    timeout = 600  # Safety margin
    
    # 1. Generate initial best configuration via enumeration
    best_points, best_lines, best_score = _generate_cluster_config()
    
    if best_points is None:
        # Fallback
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        # Get top lines for this grid
        # Simple heuristic: vertical and horizontal
        lines_list = []
        for i in range(1, 7):
            lines_list.append((1, 0, -i))
        for j in range(1, 10):
            lines_list.append((0, 1, -j))
        # Add diagonals
        for c in range(2, 16):
            lines_list.append((1, 1, -c))
            lines_list.append((1, -1, c))
        # Take first 27 unique
        seen = set()
        final_lines = []
        for l in lines_list:
            key = _normalize_line(*l)
            if key not in seen:
                seen.add(key)
                final_lines.append(key)
                if len(final_lines) == 27:
                    break
        best_lines = np.array(final_lines, dtype=np.int64)
        best_score = _count_incidences(best_points, best_lines)

    # 2. Improve with local search
    time_left = timeout - 20 - (time.time() - start_time)
    if time_left > 5:
        best_points, best_lines, best_score = _local_search_improve(best_points, best_lines, max_time=time_left)
        
    # 3. Final validation
    points_out = best_points.astype(np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Ensure shapes
    if points_out.shape != (54, 2):
        # Reconstruct from list if necessary
        pts_list = list(points_out)
        if len(pts_list) < 54:
             # Pad with random
             while len(pts_list) < 54:
                 pts_list.append((random.randint(-10, 10), random.randint(-10, 10)))
        else:
             pts_list = pts_list[:54]
        points_out = np.array(pts_list, dtype=np.int64)
        
    if lines_out.shape != (27, 3):
        lines_list = list(lines_out)
        if len(lines_list) < 27:
             while len(lines_list) < 27:
                 lines_list.append((1, 0, random.randint(-10, 10)))
        else:
             lines_list = lines_list[:27]
        lines_out = np.array(lines_list, dtype=np.int64)
        
    # Check for duplicates in points
    pts_set = set([tuple(p) for p in points_out])
    if len(pts_set) < 54:
        # Replace duplicates with new random points
        i = 0
        while len(pts_set) < 54:
            p = tuple(points_out[i])
            if p in pts_set:
                # Remove and replace
                pts_set.remove(p)
                new_p = (random.randint(-10, 10), random.randint(-10, 10))
                while new_p in pts_set:
                    new_p = (random.randint(-10, 10), random.randint(-10, 10))
                points_out[i] = new_p
                pts_set.add(new_p)
            else:
                i += 1
                # Wait, the logic above is flawed. If p is in pts_set, it's a duplicate.
                # We should iterate and fix.
                pass
        
        # Simpler robust check
        unique_pts = []
        seen = set()
        for p in points_out:
            tp = tuple(p)
            if tp not in seen:
                seen.add(tp)
                unique_pts.append(tp)
        while len(unique_pts) < 54:
            new_p = (random.randint(-10, 10), random.randint(-10, 10))
            if new_p not in seen:
                seen.add(new_p)
                unique_pts.append(new_p)
        points_out = np.array(unique_pts[:54], dtype=np.int64)
        
    # Check for invalid lines
    valid_lines = []
    seen_lines = set()
    for l in lines_out:
        A, B, C = l
        if A == 0 and B == 0:
            continue
        key = _normalize_line(A, B, C)
        if key in seen_lines:
            continue
        seen_lines.add(key)
        valid_lines.append(key)
        
    while len(valid_lines) < 27:
        # Add random valid lines
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-20, 20)
        if A == 0 and B == 0:
            continue
        key = _normalize_line(A, B, C)
        if key not in seen_lines:
            seen_lines.add(key)
            valid_lines.append(key)
            
    lines_out = np.array(valid_lines[:27], dtype=np.int64)
    
    return points_out, lines_out

# EVOLVE-BLOCK-END
