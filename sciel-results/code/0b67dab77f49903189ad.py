"""
Strategy: Stochastic Local Search (Simulated Annealing)
The seed is a static 6x5 grid. This program uses a randomized search to find a configuration
of 30 points that maximizes the number of triples with the same non-zero doubled area.
It uses vectorized NumPy computations for speed and a time-guard to ensure it finishes within limits.
"""
import numpy as np
import time
import random
import itertools
from collections import Counter

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Compute the score: the maximum multiplicity of any non-zero doubled area.
    points: numpy array of shape (30, 2)
    Returns: (max_multiplicity, best_area)
    """
    n = len(points)
    if n < 3:
        return 0, 0
    
    # Compute all triple determinants
    # To be fast, we can compute areas for all pairs and then sum, but for n=30,
    # direct triple computation is O(n^3) = 27000 ops, which is trivial.
    # However, we want to maximize the count of a SINGLE area.
    
    # Vectorized area calculation for all triples is hard to do in one shot without
    # huge memory. But n=30 is small.
    # Let's use a loop for clarity and sufficient speed, or a clever numpy trick.
    
    # Trick:
    # Area(i,j,k) = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
    # Doubled Area = | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
    
    # We can compute all pairwise differences?
    # Actually, for n=30, a simple loop over triples is ~4500 iterations (30 choose 3).
    # 4500 iterations is very fast in Python if optimized, or even slower.
    # Let's try to vectorize partially.
    
    # Let's just use a list comprehension with numpy for the inner loop, or pure python.
    # Pure python loop for 4500 items is ~1ms. That's fine.
    
    areas = []
    # Precompute coordinates
    xs = points[:, 0]
    ys = points[:, 1]
    
    # To make it faster, we can precompute differences?
    # dx[i,j] = x[j] - x[i]
    # dy[i,j] = y[j] - y[i]
    # Area(i,j,k) = dx[i,j]*dy[i,k] - dy[i,j]*dx[i,k]
    
    # Let's stick to the simplest robust method first:
    # Since we are calling this in a loop, we need it to be fast.
    # 4500 determinants.
    
    # Optimization: 
    # We don't need to store all areas if we just want the max count.
    # But we need the counts to know the max.
    
    # Let's use a dictionary or Counter.
    # To speed up, we can avoid creating a list of 4500 items and instead update a Counter.
    
    counter = Counter()
    
    # Use itertools.combinations for indices
    # But indexing into numpy arrays in a loop is slow.
    # Let's convert to lists? No, numpy scalar access is okay-ish, but let's try to be smart.
    
    # Alternative: 
    # Generate all pairs (i,j) with i<j.
    # Then for each k, compute area? No.
    
    # Let's try the direct loop with local variables for speed.
    for i in range(n-2):
        xi, yi = points[i]
        for j in range(i+1, n-1):
            xj, yj = points[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j+1, n):
                xk, yk = points[k]
                # Determinant
                det = dx_ij * (yk - yi) - dy_ij * (xk - xi)
                adet = abs(det)
                if adet > 0:
                    counter[adet] += 1
    
    if not counter:
        return 0, 0
        
    max_count, best_area = counter.most_common(1)[0]
    return max_count, best_area

def _score_only(points):
    """Wrapper for faster scoring if we just need the value."""
    count, _ = _compute_score(points)
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe (limit is 60s)
    
    # Initial points: 6x5 grid
    # The seed is [(x,y) for x in range(6) for y in range(5)]
    # Let's start with that, or a random set?
    # Random might get stuck in local minima. Grid is a good structured start.
    # Let's start with the seed grid.
    
    points = np.array([(x, y) for x in range(6) for y in range(5)], dtype=int)
    current_score, current_best_area = _compute_score(points)
    
    best_score = current_score
    best_points = points.copy()
    best_area = current_best_area
    
    # We want to maximize the count of a SPECIFIC area.
    # Strategy:
    # 1. Pick a target area (e.g., the current best, or a common one like 1, 2, etc.)
    # 2. Move points to create more triangles of that area.
    
    # Let's try a few target areas.
    # Common areas in grids are small integers.
    
    # Simulated Annealing / Hill Climbing
    # Temperature starts high, decreases.
    
    # Since the space is small (30 points in 61x61 grid), we can do random restarts or just one long run.
    # Let's do a long run with restarts if stuck.
    
    # To be effective, we should probably focus on ONE target area at a time.
    # Which one?
    # In the 6x5 grid, what is the max multiplicity?
    # Let's check the seed score first.
    
    # Actually, let's just run a generic search that tries to increase the count of the *current* most frequent area.
    # If we switch areas, we lose progress.
    # So, fix a target area T. Maximize count of T.
    
    # Which T to pick?
    # Small areas are more likely to be frequent.
    # Let's try T = 1, 2, 3, ... up to some limit, and keep the best result overall.
    
    # Given time constraints, maybe just pick the current best area and try to improve it?
    # Or pick a random small integer?
    
    # Let's try to improve the seed's best area first.
    # If that fails, try others?
    
    # Better: Run a search for a specific target area T.
    # We can run multiple searches in parallel (sequentially) for different T.
    
    # Let's define a helper to run search for a specific T.
    
    def search_for_target(T, time_limit):
        # Start from current best points or random?
        # Starting from seed grid is good.
        p = np.array([(x, y) for x in range(6) for y in range(5)], dtype=int)
        
        # Calculate current count of T
        # We need a fast counter for a specific T.
        def count_target(p, T):
            n = len(p)
            count = 0
            for i in range(n-2):
                xi, yi = p[i]
                for j in range(i+1, n-1):
                    xj, yj = p[j]
                    dx_ij = xj - xi
                    dy_ij = yj - yi
                    for k in range(j+1, n):
                        xk, yk = p[k]
                        det = dx_ij * (yk - yi) - dy_ij * (xk - xi)
                        if abs(det) == T:
                            count += 1
            return count
        
        curr_count = count_target(p, T)
        best_count = curr_count
        best_p = p.copy()
        
        temp = 10.0
        cooling = 0.99
        
        while time.time() < time_limit:
            # Pick a random point to move
            idx = random.randint(0, 29)
            old_x, old_y = p[idx]
            
            # New random position
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
            # Ensure distinctness?
            # The problem says "distinct integer coordinate pairs".
            # We must check if (new_x, new_y) is already in p.
            # If so, skip or swap?
            # Swapping is harder. Let's just skip if duplicate.
            # Or, we can maintain a set of occupied points.
            
            # For efficiency, let's just check if it's in the array.
            # O(N) check. N=30. Fine.
            
            # Actually, moving a point to an occupied spot violates "distinct".
            # So we must ensure the new point is not at any other point's location.
            
            is_valid = True
            for i in range(30):
                if i == idx:
                    continue
                if p[i][0] == new_x and p[i][1] == new_y:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
            
            # Move
            p[idx] = [new_x, new_y]
            
            new_count = count_target(p, T)
            
            if new_count > curr_count:
                curr_count = new_count
                if curr_count > best_count:
                    best_count = curr_count
                    best_p = p.copy()
            else:
                # Accept with probability exp(-(delta)/temp)
                # Here delta = curr_count - new_count (positive means worse)
                delta = curr_count - new_count
                if delta > 0:
                    if random.random() < np.exp(-delta / temp):
                        curr_count = new_count
                    else:
                        # Revert
                        p[idx] = [old_x, old_y]
                
            temp *= cooling
            
        return best_count, best_p

    # Which targets to try?
    # The seed 6x5 grid has specific areas.
    # Let's analyze the seed quickly to pick a good target.
    # Or just try T=1, T=2, T=3, T=4, T=5.
    # T=1 is "unit area" (doubled area 1? No, doubled area 1 means area 0.5).
    # Integer coordinates -> doubled area is integer.
    # Smallest non-zero doubled area is 1.
    
    # Let's try T from 1 to 10.
    # Allocate time to each.
    # Total time ~45s.
    # 10 targets -> 4.5s each.
    
    targets = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    
    time_per_target = 4.0
    
    global_best_score = 0
    global_best_points = None
    global_best_area = 0
    
    for T in targets:
        t_start = time.time()
        count, p_best = search_for_target(T, t_start + time_per_target)
        
        if count > global_best_score:
            global_best_score = count
            global_best_points = p_best
            global_best_area = T
            
        # Check deadline
        if time.time() > deadline - 5:
            break
            
    # If we found a better configuration, return it.
    # Otherwise, return the seed.
    
    if global_best_points is not None and global_best_score > _score_only(np.array([(x, y) for x in range(6) for y in range(5)])):
        # Convert numpy array to list of tuples
        return [(int(x), int(y)) for x, y in global_best_points]
    else:
        return [(x, y) for x in range(6) for y in range(5)]

# EVOLVE-BLOCK-END
