"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from math import sqrt

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 320  # safety margin before 330s kill
    best_points = []
    best_count = 0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        random.shuffle(all_points)
        
        points = []
        dist_set = set()  # set of squared distances between all pairs
        
        for p in all_points:
            if time.time() > deadline:
                break
            
            # Check validity of p against existing points
            valid = True
            px, py = p
            dists_to_existing = []
            
            for i, (qx, qy) in enumerate(points):
                dx = px - qx
                dy = py - qy
                d2 = dx * dx + dy * dy
                dists_to_existing.append((d2, i))
                
                # Condition 2: d(P, i) must not equal any existing pairwise distance
                if d2 in dist_set:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Condition 1: All distances from P to existing points must be distinct
            seen_dists = set()
            for d2, i in dists_to_existing:
                if d2 in seen_dists:
                    valid = False
                    break
                seen_dists.add(d2)
            
            if not valid:
                continue
            
            # Add p to points
            points.append(p)
            
            # Update dist_set with new distances
            for d2, i in dists_to_existing:
                dist_set.add(d2)
            
            if len(points) > best_count:
                best_count = len(points)
                best_points = points[:]
        
        if len(points) > best_count:
            best_count = len(points)
            best_points = points[:]
    
    if not best_points:
        # Fallback: return the 1D Sidon-like construction
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        best_points = [(x, 0) for x in xs]
    
    return best_points
# EVOLVE-BLOCK-END
