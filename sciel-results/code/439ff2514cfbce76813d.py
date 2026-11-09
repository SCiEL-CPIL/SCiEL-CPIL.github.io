import numpy as np
import itertools
from collections import Counter
import random
import math

def construct():
    # Pool of candidate points
    pool = [(x, y) for x in range(-15, 16) for y in range(-15, 16)]
    
    # Initial configuration: 6x5 grid centered at 0
    init_pts = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    
    def area_of_triangle(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
        
    def init_counter(pts):
        n = len(pts)
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = area_of_triangle(pts[i], pts[j], pts[k])
                    if a > 0:
                        areas.append(a)
        return Counter(areas)

    current_pts = init_pts[:]
    area_counts = init_counter(current_pts)
    
    def get_max_info(counter):
        if not counter:
            return 0, 0
        # We want the area with the highest count
        # If tie, maybe pick the smaller area? Or doesn't matter.
        max_count = 0
        best_area = 0
        for area, count in counter.items():
            if count > max_count:
                max_count = count
                best_area = area
        return best_area, max_count

    current_best_area, current_best_count = get_max_info(area_counts)
    global_best_pts = current_pts[:]
    global_best_count = current_best_count
    global_best_area = current_best_area
    
    # Simulated Annealing with Incremental Updates
    T = 50.0
    T_min = 0.01
    cooling = 0.998
    max_iters = 50000 # Can afford more due to O(N^2) update
    
    # Precompute indices for pairs not involving i? 
    # Actually, for each i, we need pairs (j,k) where j,k != i.
    
    for it in range(max_iters):
        if T < T_min:
            break
            
        idx = random.randint(0, 29)
        old_pt = current_pts[idx]
        
        # Get candidates
        # To avoid creating duplicates, we check against current set
        # This is O(N) check.
        # We can pick a random candidate from pool and check if it's in current_pts
        # Or maintain a set of current points for O(1) lookup.
        
        # Let's maintain a set of current points for fast duplicate checking
        # But we need to update it.
        # Since we only move one point, we can just check `new_pt not in current_pts`
        # `current_pts` is a list, so `in` is O(N). N=30, so 30 ops. Fine.
        
        # Pick random new point
        # Rejection sampling
        while True:
            new_pt = random.choice(pool)
            if new_pt not in current_pts:
                break
                
        # Calculate delta in area counts
        # Triangles involving idx: (idx, j, k) for all j < k, j,k != idx
        # We need to update area_counts
        # Old areas: area(old_pt, pts[j], pts[k])
        # New areas: area(new_pt, pts[j], pts[k])
        
        # To make this faster, we can precompute the list of other points
        # other_pts = [p for i, p in enumerate(current_pts) if i != idx]
        # But we need indices j, k to be consistent? No, area is symmetric.
        
        # Let's just iterate over all pairs (j, k) from the 29 other points.
        # There are C(29, 2) = 406 pairs.
        
        # We will create a temporary dict of changes or just update the counter directly
        # and track the "best" count change?
        # Actually, we need the *new* global max count.
        # Updating the counter is easy. Finding the new max is O(A) where A is number of distinct areas.
        # Distinct areas are likely small (< 1000).
        
        # Optimization: 
        # Instead of finding max from scratch every time, we can track the top few counts.
        # But simpler: just update counter and find max. 
        # 50,000 iters * 406 ops * constant + 50,000 * O(A) find max.
        # O(A) is likely fast.
        
        # Let's try it.
        
        # 1. Decrement old areas
        # 2. Increment new areas
        # 3. Find new max
        
        # To avoid creating a new Counter object, we modify in place.
        # But we need to rollback if rejected.
        # So we store the changes.
        
        changes = [] # list of (area, delta)
        
        # Get other points
        # We can just use the list current_pts and skip idx
        others = [current_pts[j] for j in range(30) if j != idx]
        
        # Iterate over pairs of others
        # Note: The triangle (idx, j, k) is the same as (idx, k, j).
        # We just need to calculate the area for the pair of vectors from idx.
        # Vector v1 = others[j] - old_pt
        # Vector v2 = others[k] - old_pt
        # Area = |det(v1, v2)|
        
        for j_idx in range(len(others)):
            p_j = others[j_idx]
            for k_idx in range(j_idx + 1, len(others)):
                p_k = others[k_idx]
                
                # Old Area
                x1, y1 = old_pt
                x2, y2 = p_j
                x3, y3 = p_k
                old_area = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                
                # New Area
                x1n, y1n = new_pt
                new_area = abs((x2-x1n)*(y3-y1n) - (y2-y1n)*(x3-x1n))
                
                if old_area > 0:
                    area_counts[old_area] -= 1
                    if area_counts[old_area] == 0:
                        del area_counts[old_area]
                    changes.append((old_area, -1))
                
                if new_area > 0:
                    area_counts[new_area] = area_counts.get(new_area, 0) + 1
                    changes.append((new_area, 1))
        
        # Calculate new best
        if area_counts:
            # Find max count
            # We can keep a track of the top few? 
            # For now, simple max.
            new_best_area, new_best_count = max(area_counts.items(), key=lambda x: x[1])
        else:
            new_best_area, new_best_count = 0, 0
            
        delta = new_best_count - current_best_count
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            current_pts[idx] = new_pt
            current_best_area = new_best_area
            current_best_count = new_best_count
            
            if current_best_count > global_best_count:
                global_best_count = current_best_count
                global_best_pts = current_pts[:]
                global_best_area = current_best_area
        else:
            # Reject: Rollback changes
            for area, delta_c in changes:
                area_counts[area] += -delta_c
                if area_counts[area] == 0:
                    del area_counts[area]
                    
        T *= cooling

    return global_best_pts
