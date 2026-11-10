import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin
    
    best_points = []
    best_score = 0
    
    # Generate all candidate points
    candidates = [(x, y) for x in range(N) for y in range(N)]
    
    # Multiple random restarts with different orderings
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        points = []
        # For each point, store set of squared distances to other selected points
        # dist_to_point[i] = set of squared distances from points[i] to all other points in set
        dist_sets = []
        # Also track: for quick check, for each point index, the distances to all others
        # When adding new point P, for each existing point A:
        #   d_PA = dist2(P, A)
        #   Check if d_PA is in dist_sets[idx_A] (meaning some B has dist(A,B)=d_PA) -> conflict
        #   Also need to check: for each existing B, if dist(P,B) == dist(P,A) -> conflict
        # The second check: collect all d_PB for B in existing, check for duplicates
        
        for P in candidates:
            if time.time() > deadline:
                break
            
            px, py = P
            # Compute squared distances to all existing points
            d_to_existing = []
            conflict = False
            
            for i, (ax, ay) in enumerate(points):
                dx = px - ax
                dy = py - ay
                d2 = dx*dx + dy*dy
                d_to_existing.append(d2)
                
                # Check if this distance already exists between A and some other point
                # i.e., is d2 in dist_sets[i]? That means there exists B such that dist(A,B)^2 = d2
                # Then triangle P,A,B has PA = AB, so isosceles
                if d2 in dist_sets[i]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Check for duplicate distances from P to existing points
            # If dist(P,A) == dist(P,B) for A != B, then triangle P,A,B is isosceles
            if len(d_to_existing) > 1:
                if len(set(d_to_existing)) != len(d_to_existing):
                    conflict = True
            
            if conflict:
                continue
            
            # No conflict, add the point
            points.append(P)
            new_idx = len(points) - 1
            dist_sets.append(set(d_to_existing))
            
            # Update dist_sets for all existing points to include distance to new point
            for i in range(new_idx):
                dist_sets[i].add(d_to_existing[i])
        
        if len(points) > best_score:
            best_score = len(points)
            best_points = points[:]
        
        # Early termination if we've found a very good solution
        if best_score >= 30:
            break
    
    return best_points
