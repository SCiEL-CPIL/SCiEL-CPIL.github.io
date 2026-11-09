import numpy as np
import time
import random
import math
from math import gcd
from collections import defaultdict

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

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
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _count_incidences(points, lines):
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _get_all_lines_and_counts(points):
    """
    Returns a list of (count, line_tuple) for all unique lines defined by pairs of points.
    """
    n = len(points)
    line_counts = defaultdict(int)
    
    # Map line key to indices of points on that line? No, just count.
    # We need to count incidences for ALL lines defined by pairs.
    # There are at most N*(N-1)/2 = 1431 pairs.
    # But multiple pairs can define the same line.
    # We need to count how many points lie on each unique line.
    
    unique_lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = _line_key(A, B, C)
            if key not in unique_lines:
                unique_lines[key] = 0
                
    # Now, for each unique line, count incidences with all points
    # This is O(L * N) where L is number of unique lines.
    # L is at most 1431. N is 54. 1431 * 54 ~ 77k ops. Very fast.
    
    results = []
    for key in unique_lines.keys():
        A, B, C = key
        cnt = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt >= 2: # Only lines with at least 2 points are candidates for "best"
            results.append((cnt, key))
            
    # Sort by count descending
    results.sort(key=lambda x: x[0], reverse=True)
    return results

def _best_lines_for_points(points, k=27):
    all_lines = _get_all_lines_and_counts(points)
    if len(all_lines) < k:
        # Should not happen with 54 distinct points
        # Pad with dummy lines? No, just return what we have.
        # But we need exactly 27.
        # If we don't have enough, we are in trouble.
        # However, 54 points generally create many lines.
        pass
        
    top_k = all_lines[:k]
    lines_arr = np.array([line for _, line in top_k], dtype=np.int64)
    
    # If we have less than 27, we need to add more.
    # This is unlikely but let's handle it by adding arbitrary lines?
    # No, the problem requires valid distinct lines.
    # If we run out of lines defined by pairs, we can't form valid lines?
    # Actually, any line is valid as long as it's distinct and A,B not both 0.
    # But lines not passing through any points contribute 0 to incidence.
    # So picking them is better than nothing if we are forced, but we won't be forced.
    
    return lines_arr

def _generate_grid_points(rows, cols, x_start=0, y_start=0, dx=1, dy=1):
    points = []
    for i in range(rows):
        for j in range(cols):
            x = x_start + i * dx
            y = y_start + j * dy
            points.append((x, y))
    return np.array(points, dtype=np.int64)

def _generate_hexagonal_points(radius):
    # Center at (0,0)
    points = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                x = q + r // 2
                y = (3 * r + r % 2) / 2 # This is for axial coords to pixel?
                # Let's use a simpler hex grid generation
                pass
    
    # Simpler hex grid:
    # (x, y) where x+y is even? Or just offset rows.
    # Row 0: 0, 1, 2...
    # Row 1: 0.5, 1.5... -> shift by 0.5? No, integers required.
    # Let's use a standard integer hex grid:
    # (x, y) such that x+y is even is not a hex grid, it's a checkerboard.
    # Hex grid in integer coords:
    # For row y, x can be -y//2 to y//2?
    # Let's just use a rectangular grid with some shear?
    # Or just stick to rectangular grids and arithmetic progressions.
    
    return np.array([], dtype=np.int64)

def _generate_arithmetic_progression_points(n, start, step_x, step_y):
    points = []
    for i in range(n):
        x = start[0] + i * step_x
        y = start[1] + i * step_y
        points.append((x, y))
    return np.array(points, dtype=np.int64)

def _local_search_lines(points, current_lines, max_iters=1000, deadline=None):
    """
    Try to improve the set of 27 lines for a fixed set of points.
    Since the optimal lines are the top 27 by incidence, this function is actually
    just re-computing the top 27. But maybe we can swap one line for another?
    No, the top 27 is always optimal for fixed points.
    So this function is just _best_lines_for_points.
    """
    return _best_lines_for_points(points)

def _local_search_points(points, lines, max_iters=50, deadline=None):
    """
    Local search on points.
    Move one point to a nearby location, recompute best lines, check score.
    """
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidences(points, lines)
    
    n = 54
    # Neighborhood: shift point by dx, dy in [-2, 2]
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        idx = random.randint(0, n - 1)
        # Try all small moves
        best_move = None
        best_move_score = best_score
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                
                new_points = best_points.copy()
                new_points[idx][0] += dx
                new_points[idx][1] += dy
                
                # Check validity (no duplicates)
                pt_set = set()
                valid = True
                for p in new_points:
                    key = (int(p[0]), int(p[1]))
                    if key in pt_set:
                        valid = False
                        break
                    pt_set.add(key)
                
                if not valid:
                    continue
                
                new_lines = _best_lines_for_points(new_points)
                if not _is_valid(new_points, new_lines):
                    continue
                    
                new_score = _count_incidences(new_points, new_lines)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (new_points, new_lines)
        
        if best_move:
            best_points, best_lines = best_move
            best_score = best_move_score
            improved = True
            
        if not improved:
            break
            
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # 1. Try various rectangular grids
    # Total points = 54
    # Factors of 54: 1, 2, 3, 6, 9, 18, 27, 54
    # Shapes: 1x54, 2x27, 3x18, 6x9, 9x6, 18x3, 27x2, 54x1
    # We can also scale them (dx, dy != 1)
    
    candidate_configs = []
    
    # Standard grids
    for rows in [1, 2, 3, 6, 9, 18, 27, 54]:
        cols = 54 // rows
        # Try different scales
        for scale in [1, 2, 3]:
            pts = _generate_grid_points(rows, cols, 0, 0, scale, scale)
            candidate_configs.append(pts)
            
            # Try sheared grids?
            # x = i + j, y = i
            # This creates a parallelogram
            pts_shear = np.array([(i + j, i) for i in range(rows) for j in range(cols)], dtype=np.int64)
            candidate_configs.append(pts_shear)
            
            # Try another shear
            pts_shear2 = np.array([(i, i + j) for i in range(rows) for j in range(cols)], dtype=np.int64)
            candidate_configs.append(pts_shear2)

    # 2. Try "Two parallel lines" or "Three parallel lines" structures?
    # e.g. 18 points on y=0, 18 on y=1, 18 on y=2
    # This is covered by 3x18 grid.
    
    # 3. Try random structured sets?
    # Maybe a set of points on a circle? No, integer coords.
    
    # Evaluate all candidates
    for pts in candidate_configs:
        if time.time() > deadline:
            break
            
        if not _is_valid(pts, np.zeros((27, 3), dtype=np.int64)):
            continue
            
        lines = _best_lines_for_points(pts)
        if not _is_valid(pts, lines):
            continue
            
        score = _count_incidences(pts, lines)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
            
    # If no valid candidate found (should not happen), fallback to simple grid
    if best_points is None:
        best_points = _generate_grid_points(6, 9, 0, 0, 1, 1)
        best_lines = _best_lines_for_points(best_points)
        best_score = _count_incidences(best_points, best_lines)
        
    # 4. Local Refinement
    # Perform local search on the best candidate found
    # We can also try local search on other top candidates
    
    # Let's keep the top 5 candidates and refine them
    # For simplicity, just refine the best one for now, or a few others if time permits.
    
    # Let's create a list of top initial candidates
    scored_candidates = []
    for pts in candidate_configs:
        if time.time() > deadline - 100: # Leave time for refinement
            break
        if not _is_valid(pts, np.zeros((27, 3), dtype=np.int64)):
            continue
        lines = _best_lines_for_points(pts)
        if not _is_valid(pts, lines):
            continue
        score = _count_incidences(pts, lines)
        scored_candidates.append((score, pts, lines))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top 10 candidates
    for i, (score, pts, lines) in enumerate(scored_candidates[:10]):
        if time.time() > deadline - 10:
            break
            
        # Local search on points
        new_pts, new_lines, new_score = _local_search_points(pts, lines, max_iters=200, deadline=deadline-10)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_pts
            best_lines = new_lines
            
    # 5. Final Validation
    if not _is_valid(best_points, best_lines):
        # Fallback
        best_points = _generate_grid_points(6, 9, 0, 0, 1, 1)
        best_lines = _best_lines_for_points(best_points)
        
    return best_points, best_lines

# To ensure we don't time out, we add a global timeout check in the main loop logic
# The _local_search_points has a deadline check.
# The main loop checks time before starting refinement.

# One optimization: _get_all_lines_and_counts is called frequently.
# It's O(N^3) effectively (N^2 pairs * N points).
# For N=54, this is ~150k ops. In Python, this might take 10-20ms.
# If we do this 1000 times, that's 10-20 seconds.
# Local search does ~25 moves per iter (5x5 grid).
# 200 iters * 25 moves = 5000 evaluations.
# 5000 * 15ms = 75 seconds.
# This might be too slow for 10 candidates.
# We need to speed up the evaluation.

# Optimization: Vectorize the incidence count for a fixed set of lines?
# No, the lines change.
# But we can speed up _get_all_lines_and_counts.
# Instead of checking all points for each line, we can use a dictionary.
# For each point, determine which lines it lies on? No, that's harder.
# The current method is:
# 1. Find all unique lines defined by pairs. (O(N^2))
# 2. For each line, count points. (O(L * N))
# L is at most N(N-1)/2.
# So O(N^3).
# Can we do better?
# For each point, we don't know the lines it's on until we check.
# However, we can pre-compute the lines through each point?
# For point i, lines are defined by (i, j) for j != i.
# This is still O(N^2) lines per point, total O(N^3).
# But we can avoid the inner loop of checking all points for each line.
# If we build the set of lines from pairs, we already know at least 2 points are on it.
# We just need to check the remaining N-2 points.
# This doesn't change the complexity class, but reduces constant.
# Current code checks all N points.
# Let's try to optimize the counting step.

def _get_all_lines_and_counts_optimized(points):
    n = len(points)
    # Map line key to set of point indices?
    # Or just count.
    # We can use a dictionary: line_key -> count
    # But we need to initialize count to 0, then add 1 for each point on the line.
    # How to find points on a line?
    # If we iterate over points, and for each point, iterate over other points,
    # we generate the line. Then we know point i and j are on it.
    # But we need to count ALL points on it.
    
    # Alternative:
    # 1. Generate all unique lines and store them in a list.
    # 2. For each line, count incidences.
    
    # Let's try to vectorize the counting for a specific line.
    # But we have many lines.
    
    # Let's stick to the Python loop but optimize it.
    # Use local variables for speed.
    
    line_counts = {}
    
    # Generate unique lines
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = _line_key(A, B, C)
            if key not in line_counts:
                line_counts[key] = 0
                
    # Count incidences
    # To speed up, we can check points in a batch?
    # No, simple loop is probably fine for N=54.
    # 1431 lines * 54 points = 77k checks.
    # Each check is 2 mults, 2 adds, 1 compare.
    # ~5 ops. 400k ops.
    # Python does ~10M simple ops/sec?
    # 400k / 10M = 0.04s.
    # So 1000 iterations = 40s.
    # This is acceptable if we limit iterations.
    
    results = []
    for key, _ in line_counts.items():
        A, B, C = key
        cnt = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt >= 2:
            results.append((cnt, key))
            
    results.sort(key=lambda x: x[0], reverse=True)
    return results

# Update _best_lines_for_points to use optimized version
def _best_lines_for_points(points, k=27):
    all_lines = _get_all_lines_and_counts_optimized(points)
    if len(all_lines) < k:
        pass
    top_k = all_lines[:k]
    lines_arr = np.array([line for _, line in top_k], dtype=np.int64)
    return lines_arr

# Re-run the main function logic with this optimized function.
# The previous definition of _best_lines_for_points was overwritten?
# No, I defined it again. Let's make sure the final code uses the optimized one.

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590
    
    best_points = None
    best_lines = None
    best_score = 0
    
    candidate_configs = []
    
    # Generate candidates
    for rows in [1, 2, 3, 6, 9, 18, 27, 54]:
        cols = 54 // rows
        for scale in [1, 2, 3]:
            pts = _generate_grid_points(rows, cols, 0, 0, scale, scale)
            candidate_configs.append(pts)
            
            pts_shear = np.array([(i + j, i) for i in range(rows) for j in range(cols)], dtype=np.int64)
            candidate_configs.append(pts_shear)
            
            pts_shear2 = np.array([(i, i + j) for i in range(rows) for j in range(cols)], dtype=np.int64)
            candidate_configs.append(pts_shear2)

    # Evaluate candidates
    scored_candidates = []
    for pts in candidate_configs:
        if time.time() > deadline - 200:
            break
        if not _is_valid(pts, np.zeros((27, 3), dtype=np.int64)):
            continue
        lines = _best_lines_for_points(pts)
        if not _is_valid(pts, lines):
            continue
        score = _count_incidences(pts, lines)
        scored_candidates.append((score, pts, lines))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    if scored_candidates:
        best_score, best_points, best_lines = scored_candidates[0]
    else:
        # Fallback
        best_points = _generate_grid_points(6, 9, 0, 0, 1, 1)
        best_lines = _best_lines_for_points(best_points)
        best_score = _count_incidences(best_points, best_lines)
        
    # Local Refinement on top candidates
    # Limit to top 5 to ensure we finish in time
    for i in range(min(5, len(scored_candidates))):
        if time.time() > deadline - 10:
            break
            
        score, pts, lines = scored_candidates[i]
        # Only refine if it's close to best or best itself
        if score < best_score * 0.9:
            continue
            
        new_pts, new_lines, new_score = _local_search_points(pts, lines, max_iters=100, deadline=deadline-10)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_pts
            best_lines = new_lines
            
    if not _is_valid(best_points, best_lines):
        best_points = _generate_grid_points(6, 9, 0, 0, 1, 1)
        best_lines = _best_lines_for_points(best_points)
        
    return best_points, best_lines
