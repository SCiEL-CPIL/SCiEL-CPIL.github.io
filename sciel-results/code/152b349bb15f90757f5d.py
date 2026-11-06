import time
import random
import math
from itertools import combinations
import numpy as np

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid).
    """
    n = len(points)
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
            count = 0
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    count += 1
                    indices_on_line.add(k)
            if count >= 5:
                return -1
            if count == 4:
                line_id = tuple(sorted(indices_on_line))
                lines_found.add(line_id)
    return len(lines_found)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # 1. Generate candidate 4-point lines
    # We use a grid of size -5..5 (11x11 = 121 points) to ensure density and integer coordinates
    # This is small enough to compute all 4-point subsets quickly with NumPy
    grid_size = 11
    offset = 5
    points_grid = [(x - offset, y - offset) for x in range(grid_size) for y in range(grid_size)]
    N_grid = len(points_grid)
    
    # Convert to numpy array for vectorization
    pts_arr = np.array(points_grid)
    
    # Precompute all 4-combinations of indices
    # Number of combinations is C(121, 4) ~ 10 million. 
    # This might be too slow to generate in Python loops. 
    # Let's reduce grid size or use a smarter generation.
    # C(100, 4) is ~ 3.9M. C(50, 4) is ~ 230k.
    # Let's use a 10x10 grid (100 points) to be safe on time.
    grid_size = 10
    offset = 5
    points_grid = [(x - offset, y - offset) for x in range(grid_size) for y in range(grid_size)]
    N_grid = len(points_grid)
    pts_arr = np.array(points_grid)
    
    # We need to find all sets of 4 indices that are collinear.
    # Instead of checking all C(100, 4) ~ 4M combinations (which is fast in C, but slow in Python loops),
    # we can iterate over pairs (i, j), find all k such that (i, j, k) are collinear, and then form 4-sets.
    
    candidate_lines = set()
    
    # Map point tuple to index for quick lookup
    point_to_idx = {tuple(p): i for i, p in enumerate(points_grid)}
    
    # Iterate over pairs
    for i in range(N_grid):
        p1 = pts_arr[i]
        for j in range(i + 1, N_grid):
            p2 = pts_arr[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Find all other points on this line
            # Vectorized check
            # Cross product (p_k - p1) x (p2 - p1) == 0
            # (x_k - x1)*dy - (y_k - y1)*dx == 0
            diff = pts_arr - p1
            cross = diff[:, 1] * dy - diff[:, 0] * dx
            mask = (cross == 0)
            
            # Get indices of points on this line
            line_indices = np.where(mask)[0]
            
            # We need at least 4 points on the line to have a 4-point line
            if len(line_indices) < 4:
                continue
            
            # Extract the 4-point subsets from these indices
            # Note: We want unique lines of 4 points.
            # If a line has 5 points, any 4 of them form a "4-point line" in the sense of the set of points,
            # BUT the problem requires "no five collinear".
            # So if a line has >= 5 points in our final set, it's invalid.
            # In our candidate generation, we should only consider lines that have EXACTLY 4 points in the grid?
            # No, the grid is just a pool. The final set is 16 points.
            # A "line" in the final set is defined by the 4 points.
            # If 5 points in the final set are collinear, it's invalid.
            
            # Strategy:
            # 1. Identify all 4-subsets of grid points that are collinear.
            # 2. Filter out any 4-subset where a 5th grid point also lies on the same line?
            #    No, that's not right. The 5th point might not be in the final set.
            #    However, to maximize the chance of a valid final set, we should prefer lines that are "sparse" in the grid?
            #    Or just collect all 4-collinear subsets.
            
            # Let's collect all 4-subsets of indices that are collinear.
            # To avoid duplicates, we only add if the sorted tuple is not in candidate_lines.
            
            # Optimization: If the line has many points, generating all C(k, 4) subsets is expensive.
            # But grid is 10x10, max collinear is 10. C(10, 4) = 210.
            # Number of pairs is ~5000. 5000 * 210 is ~ 1M operations. This is fine.
            
            # Actually, we can just add the set of all points on the line if len == 4?
            # No, we need 4-point lines.
            
            # Let's just add the specific 4-combinations.
            # To be efficient, we only need to generate combinations if the line has exactly 4 points?
            # If it has 5, any 4 are valid "4-point lines" for the purpose of building the set,
            # as long as we don't pick a 5th one later.
            
            # Let's store the frozenset of indices.
            # To reduce memory, we can store the sorted tuple of indices.
            
            # Efficiently get 4-combinations
            # Using itertools.combinations on the numpy array indices
            indices_list = [int(x) for x in line_indices]
            
            # If there are more than 4 points, we have multiple 4-point lines on the same geometric line.
            # We should add them all?
            # Yes, because any 4 of them can be part of the final set.
            
            # However, adding all of them might be redundant for the greedy step.
            # The greedy step will pick a line, add its points, and mark conflicts.
            
            # Let's add all 4-combinations.
            for comb in combinations(indices_list, 4):
                candidate_lines.add(comb)
                
    # 2. Greedy Construction
    # We have a pool of candidate 4-point lines (tuples of 4 grid indices).
    # We want to select a set of these lines such that:
    # a) The union of points is exactly 16 (or we pick 16 points from the union).
    # b) No 5 points are collinear.
    # c) Maximize the number of selected lines.
    
    # Since we want exactly 16 points, and each line has 4 points,
    # if we pick 4 disjoint lines, we get 16 points and 4 lines.
    # If we pick lines that share points, we get fewer unique points.
    # To get 16 unique points with 4-point lines, we likely want disjoint lines or lines that share points but we fill the rest with random points?
    # The task says "Maximize the number of lines containing exactly four of the points".
    # It does NOT require that all 16 points are part of a 4-point line.
    # But points not in any 4-point line don't contribute to the score.
    # So we want to pack as many 4-point lines as possible into 16 points.
    
    # Max possible lines:
    # If we have 4 disjoint lines: 4 lines, 16 points.
    # If we have lines sharing points:
    # e.g., 2 lines sharing 2 points: 4+4-2 = 6 points, 2 lines.
    # To get 16 points, we need more lines.
    
    # Let's use a greedy set packing approach.
    # 1. Shuffle candidate lines.
    # 2. Maintain a set of selected lines and a set of selected points.
    # 3. Maintain a count of points on each geometric line (to ensure no 5 collinear).
    #    Actually, we need to track the geometric line itself.
    #    Two different 4-point lines might be on the same geometric line (if the grid had 5+ points on it).
    #    If we select two 4-point lines on the same geometric line, they share 3 points (if they are from a 5-point line).
    #    Union of points: 5 points. This would create a 5-collinear set. INVALID.
    #    So we must ensure that for any geometric line, we select at most one 4-point subset?
    #    Or rather, the total number of points from the final set on any geometric line must be <= 4.
    
    # Refined Greedy:
    # - selected_points: set of grid indices
    # - line_counts: dict mapping (geometric_line_id) -> count of selected points on that line
    #   Geometric line id can be the sorted tuple of all grid points on that line? 
    #   Or a canonical representation (e.g., normalized dx, dy and a point).
    
    # Let's precompute the geometric line ID for each candidate 4-point line.
    # For a 4-point line (indices), find the full set of grid points on that line.
    # Use that full set (sorted) as the key.
    
    line_id_map = {}
    for line in candidate_lines:
        # Find all grid points collinear with these 4
        # Pick first two points
        p1 = pts_arr[line[0]]
        p2 = pts_arr[line[1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        diff = pts_arr - p1
        cross = diff[:, 1] * dy - diff[:, 0] * dx
        mask = (cross == 0)
        full_indices = tuple(sorted(np.where(mask)[0]))
        
        line_id_map[line] = full_indices
        
    # Now we have candidate_lines (set of 4-tuples) and line_id_map (4-tuple -> full-line-tuple)
    
    # Greedy selection
    best_points_set = set()
    best_lines = []
    best_score = 0
    
    # Try multiple random shuffles to escape local optima
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
            
        lines_list = list(candidate_lines)
        random.shuffle(lines_list)
        
        current_points = set()
        current_lines = []
        current_line_counts = {} # full_line_id -> count
        
        for line in lines_list:
            # Check if adding this line keeps points <= 16
            if len(current_points) + len(line) > 16:
                continue
                
            # Check if adding this line creates 5 collinear points
            # The line belongs to full_line_id
            fid = line_id_map[line]
            
            # Count how many of the points in 'line' are already in current_points
            # And how many total points on 'fid' are in current_points
            # Actually, current_line_counts[fid] is the number of selected points on that geometric line.
            
            new_points_on_line = 0
            for p in line:
                if p not in current_points:
                    new_points_on_line += 1
            
            total_on_line = current_line_counts.get(fid, 0) + new_points_on_line
            
            if total_on_line > 4:
                continue
                
            # Also, we must ensure that the union of points doesn't exceed 16.
            # We already checked len(current_points) + len(line) > 16?
            # No, that's an overestimate because of overlaps.
            # Let's calculate exact new size.
            new_size = len(current_points | set(line))
            if new_size > 16:
                continue
            
            # If valid, add it
            current_points.update(line)
            current_lines.append(line)
            current_line_counts[fid] = total_on_line
            
            # Stop if we have 16 points
            if len(current_points) == 16:
                break
        
        # Convert current_points to list of [x, y]
        if len(current_points) == 16:
            pts_list = [[int(pts_arr[i][0]), int(pts_arr[i][1])] for i in current_points]
            
            # Validate and score
            score = _count_lines_exactly_4(pts_list)
            if score > best_score:
                best_score = score
                best_points_set = current_points
                best_lines = current_lines
                if best_score >= 15: # Theoretical max
                    break

    # If we didn't get 16 points, or score is 0, try a fallback
    if len(best_points_set) != 16:
        # Fallback: 4x4 grid
        best_points_set = set()
        for x in range(-2, 2):
            for y in range(-2, 2):
                # Map back to grid index?
                # Grid is -5..5. -2..1 is within.
                idx = (x + offset) * grid_size + (y + offset)
                best_points_set.add(idx)
        
        # Ensure we have 16
        # 4x4 = 16.
        
        # Validate
        pts_list = [[int(pts_arr[i][0]), int(pts_arr[i][1])] for i in best_points_set]
        score = _count_lines_exactly_4(pts_list)
        if score > best_score:
            best_score = score

    # Final conversion
    result = []
    for i in best_points_set:
        result.append([int(pts_arr[i][0]), int(pts_arr[i][1])])
        
    # Ensure exactly 16 points
    if len(result) != 16:
        # This should not happen if logic is correct, but just in case
        # Pad with random points? Or truncate?
        # Truncating breaks the "exactly 16" if we had more, but we enforced <= 16.
        # If we have less, we need to add points.
        # This fallback is weak.
        pass
        
    return result

# Note: The above logic for "current_line_counts" might be slightly flawed if multiple candidate lines map to the same fid 
# and we add points incrementally. 
# Example: Line A (4 pts) on fid. Line B (4 pts) on fid.
# If we add A, count becomes 4.
# If we try to add B, new_points_on_line will be 0 (if B is subset of A? No, B is a different 4-comb from the same 5+ point line).
# If the line has 5 points {1,2,3,4,5}.
# Candidate A: {1,2,3,4}. Candidate B: {1,2,3,5}.
# Add A: current_points={1,2,3,4}. count_fid=4.
# Try B: new_points_on_line = 1 (point 5). total_on_line = 4 + 1 = 5. Reject. Correct.
# What if A={1,2,3,4} and B={2,3,4,5}?
# Add A: count=4.
# Try B: new_points_on_line = 1 (point 5). total=5. Reject. Correct.

# The logic holds.

if __name__ == "__main__":
    pts = construct()
    print(_count_lines_exactly_4(pts))
