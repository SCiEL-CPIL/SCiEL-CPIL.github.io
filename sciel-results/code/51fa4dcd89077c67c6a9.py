# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _canon_line(A, B, C):
    """Canonicalize a line (A, B, C) by dividing by gcd and ensuring first nonzero is positive."""
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

def _line_intersection(l1, l2):
    """Return integer intersection point of two lines, or None if parallel or non-integer."""
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # Parallel
    x_num = B1 * C2 - B2 * C1
    y_num = A2 * C1 - A1 * C2
    if x_num % det != 0 or y_num % det != 0:
        return None  # Non-integer
    x = x_num // det
    y = y_num // det
    return (x, y)

def _count_incidences(points, lines):
    """Count exact incidences using int64 arithmetic."""
    count = 0
    for (x, y) in points:
        for (A, B, C) in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count

def _generate_line_pool(max_coeff=30):
    """Generate a pool of lines with small integer coefficients."""
    lines = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_coeff, max_coeff + 1):
                canon = _canon_line(A, B, C)
                if canon is not None:
                    lines.add(canon)
    return list(lines)

def _greedy_line_selection(pool, num_lines=27, max_iter=5000):
    """
    Greedily select lines that maximize the number of integer intersection points
    with already selected lines. This creates a dense web of integer points.
    """
    selected = []
    # Start with a few random lines to break symmetry
    import random
    random.seed(42)
    pool_copy = pool[:]
    random.shuffle(pool_copy)
    
    # Select first few lines randomly but ensure they intersect at integer points
    for _ in range(3):
        if not pool_copy:
            break
        # Pick a line that intersects with existing selected lines at integer points
        best_line = None
        best_score = -1
        for l in pool_copy:
            score = 0
            for sel in selected:
                pt = _line_intersection(l, sel)
                if pt is not None:
                    score += 1
            if score > best_score:
                best_score = score
                best_line = l
        if best_line is not None:
            selected.append(best_line)
            pool_copy.remove(best_line)
        else:
            selected.append(pool_copy.pop())
    
    # Now greedily add lines that maximize new integer intersection points
    for _ in range(num_lines - len(selected)):
        best_line = None
        best_new_pts = -1
        best_total_score = -1
        
        # Build set of existing intersection points for scoring
        existing_pts = set()
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                pt = _line_intersection(selected[i], selected[j])
                if pt is not None:
                    existing_pts.add(pt)
        
        for l in pool_copy:
            # Count new integer intersection points with selected lines
            new_pts = 0
            for sel in selected:
                pt = _line_intersection(l, sel)
                if pt is not None and pt not in existing_pts:
                    new_pts += 1
            
            # Score: prioritize lines that create many new integer points
            # Also consider multiplicity: points where >=3 lines meet are better
            score = new_pts * 10 + len(selected)  # Prefer lines that connect to many existing
            
            if score > best_total_score or (score == best_total_score and new_pts > best_new_pts):
                best_total_score = score
                best_new_pts = new_pts
                best_line = l
        
        if best_line is not None:
            selected.append(best_line)
            pool_copy.remove(best_line)
        else:
            break
    
    return selected

def _find_best_points(lines, num_points=54):
    """
    Find the num_points integer points that maximize incidences with the given lines.
    Use greedy: repeatedly pick the point with highest multiplicity.
    """
    # Collect all integer intersection points
    point_lines = {}  # point -> list of line indices
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                if pt not in point_lines:
                    point_lines[pt] = []
                point_lines[pt].append(i)
                point_lines[pt].append(j)
    
    # Also add points that lie on a single line? No, we want high multiplicity.
    # But we might need more points than intersections. Let's also consider
    # points on individual lines that are not intersections.
    
    # For now, just use intersection points. If we need more, we'll extend lines.
    
    # Sort by multiplicity (number of lines passing through)
    # Note: a point from intersection of lines i and j has at least those 2 lines.
    # But it might lie on more lines. Let's verify.
    
    verified_points = {}
    for pt, line_idx_list in point_lines.items():
        # Verify which lines actually pass through this point
        x, y = pt
        actual_lines = []
        for idx in range(len(lines)):
            A, B, C = lines[idx]
            if A * x + B * y + C == 0:
                actual_lines.append(idx)
        verified_points[pt] = actual_lines
    
    # Sort by number of lines
    sorted_pts = sorted(verified_points.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Greedily select points
    selected_points = []
    used_lines_counts = [0] * len(lines)  # Track how many points are on each line
    
    for pt, line_idxs in sorted_pts:
        if len(selected_points) >= num_points:
            break
        selected_points.append(pt)
        for idx in line_idxs:
            used_lines_counts[idx] += 1
    
    # If we don't have enough points, we need to find more.
    # This can happen if there aren't enough distinct integer intersection points.
    # In that case, we might need to adjust the line selection or allow points
    # that are not at intersections.
    
    if len(selected_points) < num_points:
        # Find additional points on lines
        # For each line, find integer points on it that are not already selected
        additional = []
        for idx, (A, B, C) in enumerate(lines):
            # Find integer points on this line
            # If A != 0: x = t, y = (-A*t - C)/B if B != 0
            # Simplify: just scan a range
            found = 0
            for x in range(-100, 101):
                # Check if there's an integer y on this line
                if B != 0:
                    num = -A * x - C
                    if num % B == 0:
                        y = num // B
                        pt = (x, y)
                        if pt not in selected_points and pt not in additional:
                            # Check if this point is not already an intersection point
                            # (it might be, but we already selected those)
                            additional.append(pt)
                            found += 1
                            if found >= 10:  # Add up to 10 per line
                                break
                elif A != 0:
                    num = -C
                    if num % A == 0:
                        x_val = num // A
                        for y in range(-100, 101):
                            pt = (x_val, y)
                            if pt not in selected_points and pt not in additional:
                                additional.append(pt)
                                found += 1
                                if found >= 10:
                                    break
                    break
                else:
                    # Horizontal line: y = -C/A
                    if A != 0:
                        num = -C
                        if num % A == 0:
                            y_val = num // A
                            for x in range(-100, 101):
                                pt = (x, y_val)
                                if pt not in selected_points and pt not in additional:
                                    additional.append(pt)
                                    found += 1
                                    if found >= 10:
                                        break
        
        # Add additional points until we have enough
        for pt in additional:
            if len(selected_points) >= num_points:
                break
            selected_points.append(pt)
    
    return selected_points[:num_points]

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences using dual-line-construction:
    1. Select 27 lines that maximize integer intersection points.
    2. Place 54 points at the highest-multiplicity intersection points.
    """
    start_time = time.time()
    deadline = 600.0  # 630s limit, 30s safety margin
    
    # Generate line pool
    pool = _generate_line_pool(max_coeff=20)
    
    # Greedily select lines
    lines = _greedy_line_selection(pool, num_lines=27)
    
    # Find best points
    points = _find_best_points(lines, num_points=54)
    
    # Convert to numpy arrays
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    
    # Verify shapes
    assert points_arr.shape == (54, 2), f"Points shape: {points_arr.shape}"
    assert lines_arr.shape == (27, 3), f"Lines shape: {lines_arr.shape}"
    
    # Count incidences
    I = _count_incidences(points, lines)
    
    # If we have time, try to improve with local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:  # 60s safety margin
        # Simple local search: try swapping points
        import random
        random.seed(42)
        
        best_I = I
        best_points = points_arr.copy()
        best_lines = lines_arr.copy()
        
        # Try moving points to nearby integer coordinates
        for _ in range(1000):
            if time.time() - start_time > deadline - 30:
                break
            
            # Pick a random point and try to move it
            idx = random.randint(0, 53)
            x, y = points[idx]
            
            # Try nearby points
            best_move = None
            best_move_I = best_I
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (x + dx, y + dy)
                    
                    # Check if this point is already used
                    if any(p[0] == new_pt[0] and p[1] == new_pt[1] for p in points if points.index(p) != idx):
                        continue
                    
                    # Count incidences for new point
                    new_I = best_I
                    # Remove old incidences
                    for (A, B, C) in lines:
                        if A * x + B * y + C == 0:
                            new_I -= 1
                        if A * new_pt[0] + B * new_pt[1] + C == 0:
                            new_I += 1
                    
                    if new_I > best_move_I:
                        best_move_I = new_I
                        best_move = new_pt
            
            if best_move is not None:
                points[idx] = best_move
                best_I = best_move_I
                best_points[idx] = best_move
        
        if best_I > I:
            points = best_points.tolist()
            I = best_I
    
    # Final conversion
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
