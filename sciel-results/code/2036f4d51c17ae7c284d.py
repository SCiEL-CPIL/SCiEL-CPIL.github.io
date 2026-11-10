import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0

    def score(pts):
        n = len(pts)
        if n < 3:
            return 0
        arr = np.array(pts, dtype=np.int64)
        counts = Counter()
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                xk, yk = rest[:, 0], rest[:, 1]
                # 2*Area = |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
                doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += int(c)
        if not counts:
            return 0
        return max(counts.values())

    # Strategy:
    # 1. Generate a dense set of candidate points in a central region.
    #    Points closer together tend to form smaller area triangles.
    #    We focus on a grid where small integer differences are common.
    
    # 2. Pre-compute "yield" for each candidate point:
    #    How many other candidate points form a triangle of area 1 or 2 with it?
    #    (We can't know the final set, so we approximate with the full candidate pool).
    
    # 3. Greedily select 30 points with highest yield, then refine with local search.

    # Candidate pool: A 10x10 grid in the center, plus some jittered points.
    # Let's try a grid of size 6x5 = 30 points exactly? No, we need to select from a larger pool.
    # Let's use a 12x12 grid centered at 0. That's 144 points.
    # We want to pick 30 from these 144 to maximize small area triangles.
    
    grid_size = 12
    candidates = []
    for x in range(-grid_size//2, grid_size//2 + 1):
        for y in range(-grid_size//2, grid_size//2 + 1):
            candidates.append((x, y))
            
    # Add some points slightly outside to allow for better local search moves later
    # But for the initial greedy selection, let's stick to the dense grid.
    # Actually, let's make the grid a bit larger to have more choice, say 15x15 = 225 points.
    grid_size = 15
    candidates = []
    for x in range(-grid_size//2, grid_size//2 + 1):
        for y in range(-grid_size//2, grid_size//2 + 1):
            candidates.append((x, y))
            
    cand_set = set(candidates)
    n_cands = len(candidates)
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # Pre-compute pair-wise distances or just use a heuristic yield.
    # A point's yield is roughly proportional to the number of other points it can form
    # small area triangles with. 
    # For a grid, interior points are better than edge points.
    # Let's compute a simple score: number of neighbors within Chebyshev distance 2.
    
    yields = np.zeros(n_cands, dtype=np.int32)
    for i in range(n_cands):
        xi, yi = cand_arr[i]
        for j in range(n_cands):
            if i == j: continue
            xj, yj = cand_arr[j]
            dx = abs(xi - xj)
            dy = abs(yi - yj)
            # If they are close, they are more likely to form small area triangles with a third point
            # This is a proxy metric.
            if dx <= 2 and dy <= 2:
                yields[i] += 1
                
    # Sort candidates by yield
    sorted_indices = np.argsort(-yields)
    
    # Greedy selection: Pick top 30? 
    # This might result in all points being clustered in the center, which is good for small areas.
    # But we need distinct points.
    
    selected_indices = sorted_indices[:30]
    initial_pts = [candidates[i] for i in selected_indices]
    
    # Check if initial score is good
    current_score = score(initial_pts)
    best_score = current_score
    best_pts = initial_pts
    
    # Local Search:
    # Try to swap a point in the set with a point outside the set (from the candidate pool)
    # to improve the score.
    
    current_set = set(initial_pts)
    outside_set = list(cand_set - current_set)
    
    # To make local search faster, we can pre-compute the score contribution or use a delta heuristic.
    # But recomputing full score is O(N^3) ~ 27000 ops, which is fast enough in NumPy if vectorized.
    # Let's optimize the score function for local search by only checking affected triangles.
    # Actually, with N=30, full re-evaluation is ~ 30^3/6 = 4500 triangles. 
    # 4500 * (vectorized ops) is very fast.
    # Let's just do full re-evaluation for simplicity and correctness, as it's fast.
    
    def score_delta(pts, old_idx, old_pt, new_pt):
        # This is a placeholder, we'll just call score()
        pts[old_idx] = new_pt
        s = score(pts)
        pts[old_idx] = old_pt
        return s

    # Simulated Annealing / Hill Climbing
    # We have ~ 35 seconds left.
    # Let's do a series of restarts or just one long SA run.
    
    # Let's use a better initialization: 
    # Randomly pick 30 from the top 50 highest yield points.
    
    random.seed(42)
    
    for restart in range(5):
        if time.time() > total_deadline - 5:
            break
            
        # Pick 30 from top 60 yields
        top_k = 60
        candidates_indices = sorted_indices[:top_k]
        random.shuffle(candidates_indices)
        selected_indices = candidates_indices[:30]
        current_pts = [candidates[i] for i in selected_indices]
        current_set = set(current_pts)
        
        current_score = score(current_pts)
        
        # Local Search
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < total_deadline - 5:
            # Pick a random index in current_pts
            i = random.randint(0, 29)
            old_pt = current_pts[i]
            
            # Pick a random point from outside_set
            if not outside_set:
                break
            j = random.randint(0, len(outside_set) - 1)
            new_pt = outside_set[j]
            
            if new_pt == old_pt:
                continue
                
            # Swap
            current_pts[i] = new_pt
            new_score = score(current_pts)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                # Accept
                current_score = new_score
                # Update sets
                current_set.remove(old_pt)
                current_set.add(new_pt)
                outside_set.remove(new_pt)
                outside_set.append(old_pt)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = current_pts[:]
            else:
                # Reject
                current_pts[i] = old_pt
                
            T = max(T_min, T * cooling)
            
            # If we've been stuck for a while, maybe restart with a different random seed?
            # The loop condition handles the time.

    return best_pts
