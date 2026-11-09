"""Program for no-isosceles set task using bitset-based distance tracking."""
import numpy as np
import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    
    Approach: Randomized Greedy with bitset-based squared distance tracking.
    We perform multiple greedy passes with different orderings (including random shuffles)
    to escape local minima. We use bitsets to efficiently track forbidden distances.
    """
    N_local = N
    candidates = [(x, y) for x in range(N_local) for y in range(N_local)]
    
    def run_greedy(ordered_candidates):
        points = []
        dist_bitsets = []  # dist_bitsets[i] has bit j set if there's a point in points at squared distance j from points[i]
        
        for (px, py) in ordered_candidates:
            valid = True
            new_dists = []
            # We also need to check if any new_dist duplicates an existing new_dist
            # We use a bitset for this local check.
            local_dist_bitset = 0
            
            for i, (ax, ay) in enumerate(points):
                d2 = (px - ax) ** 2 + (py - ay) ** 2
                
                # Check 2: d2 must not be in dist_bitsets[i] (conflict with existing pair)
                if (dist_bitsets[i] >> d2) & 1:
                    valid = False
                    break
                
                # Check 1: d2 must not be in local_dist_bitset (conflict with another new distance)
                if (local_dist_bitset >> d2) & 1:
                    valid = False
                    break
                
                # Add to local tracking
                local_dist_bitset |= (1 << d2)
                new_dists.append(d2)
            
            if not valid:
                continue
            
            # All checks passed, add the point
            points.append((px, py))
            
            # Update dist_bitsets
            # For the new point, its dist_bitset is local_dist_bitset
            dist_bitsets.append(local_dist_bitset)
            
            # For each existing point i, add the new distance d2 to its bitset
            for i, d2 in enumerate(new_dists):
                dist_bitsets[i] |= (1 << d2)
        
        return points

    best_points = []
    start_time = time.time()
    time_limit = 280.0  # Leave 20s buffer for safety
    
    # 1. Deterministic orderings
    orderings = [
        sorted(candidates, key=lambda p: (p[0], p[1])),
        sorted(candidates, key=lambda p: (p[0] + p[1], p[0])),
        sorted(candidates, key=lambda p: (p[1], p[0])),
        sorted(candidates, key=lambda p: (max(p[0],p[1]), min(p[0],p[1]))),
    ]
    
    for order in orderings:
        if time.time() - start_time > time_limit:
            break
        pts = run_greedy(order)
        if len(pts) > len(best_points):
            best_points = pts

    # 2. Randomized orderings
    # Estimate how many random restarts we can do.
    # One pass is fast due to bitsets. Let's try 50 random restarts.
    
    rng = random.Random(42)
    for i in range(50):
        if time.time() - start_time > time_limit:
            break
        random_order = candidates.copy()
        rng.shuffle(random_order)
        pts = run_greedy(random_order)
        if len(pts) > len(best_points):
            best_points = pts

    return best_points
# EVOLVE-BLOCK-END
