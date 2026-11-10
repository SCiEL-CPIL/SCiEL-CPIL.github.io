# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
import heapq

def _canon_line(A, B, C):
    """Canonical form of a line (A, B, C) -> (a, b, c) where gcd(a,b,c)=1 and first nonzero is positive."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    """Get line (A, B, C) through two integer points, in canonical form."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
    A = dy
    B = -dx
    C = x1 * y2 - x2 * y1
    return _canon_line(A, B, C)

def _build_line_map(points):
    """
    Build a dictionary mapping canonical lines to the count of points on them.
    Returns (line_counts_dict, all_lines_list)
    """
    n = len(points)
    line_counts = {}
    
    # Generate all unique lines defined by pairs
    # To avoid O(n^2) duplicates, we can just iterate pairs and update counts
    # But we need to count how many points are on each line.
    # Efficient way: 
    # 1. Find all unique lines.
    # 2. For each point, find all lines passing through it and increment count.
    
    # Step 1: Find unique lines
    unique_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_two_points(points[i], points[j])
            if line is not None:
                unique_lines.add(line)
    
    # Step 2: Count points on each line
    # For each point, check which lines it lies on?
    # That is O(n * |lines|). |lines| ~ 1400. n=54. 54*1400 = 75k. OK.
    
    for p in points:
        x, y = p
        for line in unique_lines:
            A, B, C = line
            if A * x + B * y + C == 0:
                line_counts[line] = line_counts.get(line, 0) + 1
                
    return line_counts, list(unique_lines)

def _get_top_k_lines(line_counts, k=27):
    """Get the top k lines by count."""
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda item: item[1], reverse=True)
    return [line for line, count in sorted_lines[:k]]

def _calculate_incidence(points, lines):
    """Calculate total incidences."""
    count = 0
    for p in points:
        x, y = p
        for line in lines:
            A, B, C = line
            if A * x + B * y + C == 0:
                count += 1
    return count

def _is_valid(points, lines):
    """Check if points and lines are valid per the task contract."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for NaN/inf and integrality
    for arr in [points, lines]:
        if not np.all(np.isfinite(arr.astype(float))):
            return False
        if not np.allclose(arr, np.round(arr), atol=1e-6):
            return False
    
    points_int = np.round(points).astype(np.int64)
    lines_int = np.round(lines).astype(np.int64)
    
    # Check for repeated points
    if len(set(map(tuple, points_int))) != 54:
        return False
    
    # Check lines
    seen_lines = set()
    for i in range(27):
        A, B, C = lines_int[i]
        if A == 0 and B == 0:
            return False
        canon = _canon_line(A, B, C)
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
        
    return True

def _grid_init(k, m):
    """Generate a k x m grid of points."""
    xs = np.arange(k)
    ys = np.arange(m)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _local_search(points, deadline, max_iter=2000):
    """
    Local search: move points to improve incidence.
    Uses line count updates for efficiency.
    """
    points = points.copy()
    current_score = 0
    current_lines = []
    
    # Initial evaluation
    line_counts, all_lines = _build_line_map(points)
    current_lines = _get_top_k_lines(line_counts, 27)
    current_score = _calculate_incidence(points, current_lines)
    
    best_points = points.copy()
    best_score = current_score
    best_lines = current_lines.copy()
    
    for iter in range(max_iter):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 53)
        old_x, old_y = points[idx]
        
        # Try a few random moves
        found_improvement = False
        for _ in range(10): # Try 10 random perturbations
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = old_x + dx, old_y + dy
            
            # Check collision
            collision = False
            for i in range(54):
                if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                    collision = True
                    break
            if collision:
                continue
                
            # Move point
            points[idx] = [new_x, new_y]
            
            # Re-evaluate
            # Optimization: We could update line_counts incrementally, 
            # but for simplicity and correctness, we rebuild. 
            # 54 points, ~1400 lines. Rebuild is fast enough for 2000 iters?
            # 2000 * 75k ops = 150M ops. Might be slow in Python.
            # Let's try to be smarter or reduce iterations.
            
            line_counts_new, _ = _build_line_map(points)
            new_lines = _get_top_k_lines(line_counts_new, 27)
            new_score = _calculate_incidence(points, new_lines)
            
            if new_score > current_score:
                current_score = new_score
                current_lines = new_lines
                if new_score > best_score:
                    best_score = new_score
                    best_points = points.copy()
                    best_lines = new_lines.copy()
                found_improvement = True
                break # Accept and try next point
            else:
                # Revert
                points[idx] = [old_x, old_y]
                
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    deadline = time.time() + 590 # Safety margin
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Try different grid aspect ratios
    grids = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1)
    ]
    
    # Add some random structured starts
    for _ in range(5):
        # Random cluster
        center_x = random.randint(-20, 20)
        center_y = random.randint(-20, 20)
        pts = []
        seen = set()
        while len(pts) < 54:
            x = center_x + random.randint(-5, 5)
            y = center_y + random.randint(-5, 5)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append((x, y))
        grids.append(pts)
        
    for g in grids:
        if time.time() > deadline:
            break
            
        if isinstance(g, tuple):
            points = _grid_init(g[0], g[1])
        else:
            points = np.array(g, dtype=np.int64)
            
        # Initial line selection
        line_counts, _ = _build_line_map(points)
        lines = _get_top_k_lines(line_counts, 27)
        score = _calculate_incidence(points, lines)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
            
        # Local search
        if time.time() < deadline - 60: # Leave 60s for other runs
            p_ls, l_ls, s_ls = _local_search(points, deadline, max_iter=1000)
            if s_ls > best_score:
                best_score = s_ls
                best_points = p_ls
                best_lines = l_ls

    # Final validation and cleanup
    if best_points is None:
        best_points = _grid_init(6, 9)
        line_counts, _ = _build_line_map(best_points)
        best_lines = _get_top_k_lines(line_counts, 27)
        
    points_out = np.round(best_points).astype(np.int64)
    lines_out = np.array(best_lines, dtype=np.int64)
    
    # Ensure shapes
    if points_out.shape != (54, 2):
        raise ValueError("Invalid points shape")
    if lines_out.shape != (27, 3):
        raise ValueError("Invalid lines shape")
        
    return points_out, lines_out

# EVOLVE-BLOCK-END
