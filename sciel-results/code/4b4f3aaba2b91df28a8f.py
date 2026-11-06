import time
import random
import numpy as np
from itertools import combinations

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid) or if points are invalid.
    """
    n = len(points)
    if n != 16:
        return -1
    if len(set(tuple(p) for p in points)) != n:
        return -1
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
    
    lines_found = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    indices_on_line.add(k)
            
            if len(indices_on_line) >= 5:
                return -1
            if len(indices_on_line) == 4:
                line_id = tuple(sorted(indices_on_line))
                lines_found.add(line_id)
    return len(lines_found)

def _get_all_lines(points):
    """
    Returns a list of frozensets of point indices for all lines with >= 3 points.
    Also returns a set of all lines with >= 5 points (for invalidity check).
    """
    n = len(points)
    lines = []
    bad_lines = set()
    
    # Use a set to avoid duplicate lines (same set of indices)
    # We identify a line by the sorted tuple of indices on it.
    seen_lines = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            indices = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    indices.append(k)
            
            if len(indices) >= 3:
                idx_tuple = tuple(sorted(indices))
                if idx_tuple not in seen_lines:
                    seen_lines.add(idx_tuple)
                    lines.append(frozenset(idx_tuple))
                    if len(indices) >= 5:
                        bad_lines.add(idx_tuple)
                        
    return lines, bad_lines

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Candidate pool: A 25x25 grid centered at 0.
    # This ensures many potential collinearities.
    # We will select 16 points from this pool.
    # To allow for "non-grid" solutions if needed, we could add random points,
    # but the problem structure suggests grid-like solutions are optimal.
    # Let's use a slightly larger grid to give more flexibility, e.g., -20 to 20.
    # But 25x25 = 625 points. Picking 16 is easy.
    # Let's stick to a reasonable size to keep pair calculations fast.
    # 21x21 = 441 points.
    
    grid_size = 21
    half = grid_size // 2
    candidates = []
    for x in range(-half, half + 1):
        for y in range(-half, half + 1):
            candidates.append((x, y))
            
    candidates = np.array(candidates, dtype=np.int64)
    num_candidates = len(candidates)
    
    # Pre-compute all lines in the candidate pool?
    # That might be too expensive if we do it naively for 441 points.
    # Instead, we will compute lines on the fly for the selected subset.
    
    best_score = -1
    best_points = None
    
    # Greedy construction with randomization
    # Strategy:
    # 1. Start with an empty set of points.
    # 2. Iteratively add a point that maximizes the number of new 4-lines formed,
    #    subject to not creating any 5-lines.
    # 3. Use random restarts to escape local optima.
    
    num_restarts = 500
    points_needed = 16
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize the order of candidates to break symmetry
        cand_indices = list(range(num_candidates))
        random.shuffle(cand_indices)
        
        selected_indices = []
        selected_points = []
        
        # We need a fast way to check if adding a point creates a 5-line.
        # And to count how many 4-lines are formed.
        
        # To make this efficient, we can maintain the set of current points.
        # For each candidate point P, check against all existing points.
        
        # Optimization:
        # Instead of checking all pairs for every new point, we can just check the new point against existing points.
        # If we add P, new lines are formed by P and 2 existing points (making 3) or P and 3 existing points (making 4).
        # A 5-line is formed if P lies on a line with 4 existing points.
        
        # Let's implement a helper that checks validity and score delta for adding a point.
        
        current_pts = []
        
        for _ in range(points_needed):
            best_delta = -1
            best_candidate_idx = -1
            best_candidate_pt = None
            
            # We only need to consider candidates not already in current_pts
            # To speed up, we can just iterate through all candidates and skip those in current_pts.
            # For 441 candidates and 16 steps, this is 441*16 = ~7000 checks.
            # Each check involves checking collinearity with existing points.
            
            # To make it faster, we can pre-compute lines for the current set?
            # No, the set changes.
            
            # Let's just do the O(N_current * N_candidates) check.
            # N_current is at most 15. N_candidates is 441.
            # 15 * 441 = 6615 pair checks per step.
            # 16 steps = ~100k pair checks per restart.
            # 500 restarts = 50M checks. Might be too slow in Python if not optimized.
            
            # Optimization: 
            # We can maintain a list of "lines" (sets of indices) for the current points.
            # When adding a new point, we only need to check if it completes any line.
            
            # Let's try a simpler approach first and see if it's fast enough.
            # If not, we will optimize.
            
            # To speed up the inner loop, we can use numpy for the current points?
            # No, pure python might be faster for small N if we avoid overhead.
            
            # Let's try to optimize the "check candidate" part.
            # For a candidate P, we need to know:
            # 1. Does it form a 5-line? (i.e., are there 4 existing points collinear with P?)
            # 2. How many 4-lines does it form? (i.e., are there 3 existing points collinear with P?)
            
            # We can pre-compute for the current set of points:
            # For every pair of existing points (A, B), what is the line?
            # Actually, for a fixed P, the points collinear with P are determined by slopes.
            
            # Let's just iterate.
            
            # To avoid duplicate work, we can sort candidates by some heuristic?
            # No, random order is good for diversity.
            
            # We will iterate through all candidates.
            for ci in cand_indices:
                P = candidates[ci]
                
                # Check if P is already selected
                if P in selected_points:
                    continue
                
                # Check collinearity with existing points
                # Count how many existing points are collinear with P for each direction?
                # No, we just need to know if there's a set of 3 existing points that with P make 4,
                # or 4 existing points that with P make 5.
                
                # Let's group existing points by their slope relative to P.
                # If 3 points have the same slope as P, they form a line of 4 with P.
                # If 4 points have the same slope as P, they form a line of 5 with P.
                
                slopes = {}
                for Q in current_pts:
                    dx = P[0] - Q[0]
                    dy = P[1] - Q[1]
                    if dx == 0 and dy == 0:
                        continue # Should not happen as P not in current_pts
                    # Normalize slope (dx, dy)
                    # Use gcd to normalize direction
                    g = np.gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    ndx = dx // g
                    ndy = dy // g
                    # Ensure canonical direction (e.g., ndx > 0 or (ndx == 0 and ndy > 0))
                    if ndx < 0 or (ndx == 0 and ndy < 0):
                        ndx = -ndx
                        ndy = -ndy
                    key = (ndx, ndy)
                    if key not in slopes:
                        slopes[key] = 0
                    slopes[key] += 1
                
                # Now check slopes
                valid = True
                new_4_lines = 0
                
                for count in slopes.values():
                    # count is number of existing points on this line with P
                    total_on_line = count + 1 # +1 for P
                    if total_on_line >= 5:
                        valid = False
                        break
                    if total_on_line == 4:
                        new_4_lines += 1
                
                if not valid:
                    continue
                
                if new_4_lines > best_delta:
                    best_delta = new_4_lines
                    best_candidate_idx = ci
                    best_candidate_pt = P
            
            if best_candidate_idx == -1:
                # No valid candidate found? This shouldn't happen with a large grid.
                # If it does, break restart.
                break
                
            selected_points.append(best_candidate_pt)
            current_pts = selected_points[:]
            
        # After selecting 16 points, evaluate the full score
        if len(selected_points) == 16:
            score = _count_lines_exactly_4(selected_points)
            if score > best_score:
                best_score = score
                best_points = [p.tolist() for p in selected_points]
                # print(f"Restart {restart}: Score {best_score}")
                
        # If we found a high score, we could break early?
        # The real value is 15. If we hit 15, we can stop.
        if best_score >= 15:
            break
            
    # If we didn't find any valid points, return a known valid configuration (4x4 grid)
    if best_points is None:
        best_points = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                best_points.append([x, y])
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(_count_lines_exactly_4(pts))
