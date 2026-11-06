import time
import random
import numpy as np
from itertools import combinations

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

def _score_vectorized(pts):
    """
    Fast vectorized scoring.
    Returns:
        count_4: number of 4-collinear subsets
        count_5: number of 5-collinear subsets
    """
    n = len(pts)
    # Compute all pairwise differences
    # diff[i, j, 0] = pts[j,0] - pts[i,0]
    # diff[i, j, 1] = pts[j,1] - pts[i,1]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Cross product for collinearity check: (p_j - p_i) x (p_k - p_i) == 0
    # We want to count triples (i, j, k) such that they are collinear.
    # A set of 4 points {a,b,c,d} is collinear if all 4 pairs are collinear?
    # Actually, it's easier to count lines by their unique point sets.
    
    # Let's use a slightly different approach for speed and correctness:
    # For each pair (i, j), find all k > j such that i, j, k are collinear.
    # This is O(n^3) but with small constants.
    # However, vectorizing the "count k" part is hard.
    
    # Alternative: 
    # 1. Calculate cross products for all i, j, k.
    # cross[i, j, k] = (pts[j]-pts[i]) x (pts[k]-pts[i])
    # This is an n x n x n array. For n=16, 16^3 = 4096 elements. Very fast.
    
    # cross[i, j, k] = (pts[j,0] - pts[i,0]) * (pts[k,1] - pts[i,1]) - (pts[j,1] - pts[i,1]) * (pts[k,0] - pts[i,0])
    
    # Let's build it.
    # d_ij = pts[j] - pts[i]
    # d_ik = pts[k] - pts[i]
    
    # pts is (16, 2)
    # d_ij shape: (16, 16, 2) where d_ij[i,j] = pts[j] - pts[i]
    # d_ik shape: (16, 16, 2) where d_ik[i,k] = pts[k] - pts[i]
    
    # cross[i, j, k] = d_ij[i, j, 0] * d_ik[i, k, 1] - d_ij[i, j, 1] * d_ik[i, k, 0]
    
    d_ij = pts[np.newaxis, :, :] - pts[:, np.newaxis, :] # (16, 16, 2) -> wait, pts[j] - pts[i]
    # pts[j] is pts[:, j, :]? No, pts is (N, 2).
    # pts[np.newaxis, :, :] is (1, N, 2) -> broadcast to (N, N, 2) as pts[j]
    # pts[:, np.newaxis, :] is (N, 1, 2) -> broadcast to (N, N, 2) as pts[i]
    # So d_ij[i, j] = pts[j] - pts[i]. Correct.
    
    d_ik = pts[np.newaxis, :, :] - pts[:, np.newaxis, :] # Same as d_ij, just index k.
    
    # cross[i, j, k]
    # d_ij[i, j, 0] is shape (N, N, 1)
    # d_ik[i, k, 1] is shape (N, N, 1)
    # We need to align axes: i is axis 0, j is axis 1, k is axis 2.
    # d_ij[i, j, :] -> (N, N, 2). We want component 0 for x, 1 for y.
    # d_ik[i, k, :] -> (N, N, 2). We want component 1 for y, 0 for x.
    
    # Let's reshape to make broadcasting work for (i, j, k).
    # d_ij_x[i, j] = d_ij[i, j, 0] -> shape (N, N)
    # d_ij_y[i, j] = d_ij[i, j, 1] -> shape (N, N)
    # d_ik_x[i, k] = d_ik[i, k, 0] -> shape (N, N)
    # d_ik_y[i, k] = d_ik[i, k, 1] -> shape (N, N)
    
    d_ij_x = d_ij[:, :, 0] # (N, N)
    d_ij_y = d_ij[:, :, 1] # (N, N)
    d_ik_x = d_ik[:, :, 0] # (N, N)
    d_ik_y = d_ik[:, :, 1] # (N, N)
    
    # cross[i, j, k] = d_ij_x[i, j] * d_ik_y[i, k] - d_ij_y[i, j] * d_ik_x[i, k]
    # d_ij_x[:, :, np.newaxis] -> (N, N, 1)
    # d_ik_y[np.newaxis, :, :] -> (1, N, N)
    # Result: (N, N, N)
    
    cross = d_ij_x[:, :, np.newaxis] * d_ik_y[np.newaxis, :, :] - \
            d_ij_y[:, :, np.newaxis] * d_ik_x[np.newaxis, :, :]
    
    # cross[i, j, k] == 0 means i, j, k are collinear.
    # Note: i=j or j=k or i=k gives 0. We only care about distinct i, j, k.
    # Also, order matters for counting lines.
    
    # To count lines of exactly 4 points:
    # A line is defined by a set of points S.
    # We can iterate over all pairs (i, j) and find all k such that i, j, k collinear.
    # This is effectively what the cross product array allows.
    
    # Let's count the number of distinct lines with size 4 and 5.
    # A line with m points has C(m, 2) pairs.
    # If we just count pairs, we can't easily distinguish lines.
    
    # Better: Identify unique lines.
    # For each pair (i, j) with i < j, determine the line.
    # Then count how many points are on it.
    
    # We can do this efficiently using the cross product array.
    # For a fixed pair (i, j), the points k on the line are those where cross[i, j, k] == 0.
    
    lines_4 = 0
    lines_5 = 0
    seen_lines = set()
    
    # To avoid duplicates, we only consider pairs (i, j) where i < j.
    # And we identify the line by the set of points on it.
    
    for i in range(n):
        for j in range(i + 1, n):
            # Find all k such that cross[i, j, k] == 0
            mask = (cross[i, j, :] == 0)
            # k must be distinct from i and j? 
            # cross[i, j, i] = 0 and cross[i, j, j] = 0 always.
            # So mask will always have i and j as True.
            
            # Get indices
            idxs = np.where(mask)[0]
            
            # The line is defined by the set of points idxs.
            # To avoid counting the same line multiple times (once for each pair on it),
            # we can use a canonical representative.
            # E.g., the smallest pair (min_id, second_min_id) on the line?
            # Or just the sorted tuple of all points on the line.
            
            line_id = tuple(sorted(idxs.tolist()))
            
            if line_id in seen_lines:
                continue
                
            seen_lines.add(line_id)
            
            count = len(idxs)
            
            if count == 4:
                lines_4 += 1
            elif count == 5:
                lines_5 += 1
            # count > 5 is also bad, but lines_5 penalty covers it if we penalize per 5-subset?
            # The problem says "no five collinear". 
            # If we have 6 collinear, we have C(6,5)=6 subsets of 5.
            # The penalty should be severe enough to reject any count >= 5.
            
    return lines_4, lines_5

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n_points = 16
    
    # Initialize with a 4x4 grid
    # Points: -2, -1, 0, 1
    init_pts = []
    for x in range(-2, 2):
        for y in range(-2, 2):
            init_pts.append([x, y])
            
    current_pts = np.array(init_pts, dtype=np.int64)
    
    # Evaluate initial
    c4, c5 = _score_vectorized(current_pts)
    # Objective: Maximize c4 - penalty * c5
    # If c5 > 0, score is very low.
    penalty = 1000
    current_score = c4 - penalty * c5
    
    best_pts = current_pts.copy()
    best_score = current_score
    best_c4 = c4
    best_c5 = c5
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    max_steps = 50000 # Adjust based on speed
    
    # Pre-compute random steps? No, generate on the fly.
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        idx = random.randint(0, n_points - 1)
        
        # Pick a random move
        # Move x or y by a small amount
        # To allow larger jumps, use a distribution that allows bigger steps sometimes
        move_size = random.randint(-3, 3)
        if move_size == 0:
            continue
            
        axis = random.randint(0, 1)
        
        new_pts = current_pts.copy()
        new_pts[idx, axis] += move_size
        
        # Check bounds
        if np.any(np.abs(new_pts) > 1000):
            continue
            
        # Check duplicates
        # If any two points are identical, score is -infinity (or very low)
        # We can check this quickly
        # Sort and check diffs? Or just set size?
        # Set size check on numpy array is slow.
        # Let's assume duplicates are rare and penalize them heavily in score if they happen?
        # Actually, _score_vectorized doesn't handle duplicates well (lines might be undefined or weird).
        # Let's add a quick check.
        # Flatten and check unique count?
        # For n=16, this is fast.
        if len(set(map(tuple, new_pts))) < n_points:
            new_score = -10000
        else:
            c4_new, c5_new = _score_vectorized(new_pts)
            new_score = c4_new - penalty * c5_new
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                best_c4, best_c5 = c4_new, c5_new
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final validation and return
    # Ensure best_pts is valid
    # If best_c5 > 0, it's invalid. We need to find a valid configuration.
    # If SA didn't find a valid one, we might need to do a local search to fix collisions.
    
    result_pts = best_pts.tolist()
    
    # If the best found is invalid (c5 > 0), try to repair it
    if best_c5 > 0:
        # Simple repair: 
        # Find the line with 5+ points.
        # Move one of the points on that line to a random location.
        # Repeat.
        for _ in range(1000):
            if time.time() > deadline:
                break
            c4_check, c5_check = _score_vectorized(np.array(result_pts))
            if c5_check == 0:
                break
                
            # Find a point to move
            # Identify the bad line
            # We can just pick a random point and move it
            idx = random.randint(0, n_points - 1)
            result_pts[idx] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            
            # Ensure distinct
            # Simple check
            if len(set(map(tuple, result_pts))) < n_points:
                # Try again
                continue
                
    # Final check
    final_score = _count_lines_exactly_4(result_pts)
    
    # If final_score is -1 (invalid), we have a problem.
    # The task requires a valid result.
    # If we can't find one, we should return a known valid configuration.
    # The 4x4 grid is valid (score 4).
    if final_score < 0:
        result_pts = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                result_pts.append([x, y])
        
    return result_pts

if __name__ == "__main__":
    pts = construct()
    print(_count_lines_exactly_4(pts))
