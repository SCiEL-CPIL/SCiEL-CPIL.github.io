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
    We run the greedy algorithm multiple times with different random orderings of grid points.
    For each ordering, we iterate through points and add them if they don't create an isosceles triangle.
    """
    N_local = N
    
    # Precompute all candidate points
    candidates = [(x, y) for x in range(N_local) for y in range(N_local)]
    
    best_points = []
    best_size = 0
    
    start_time = time.time()
    time_limit = 290  # Leave 10s buffer for safety
    
    while time.time() - start_time < time_limit:
        # Random permutation
        random.shuffle(candidates)
        
        points = []
        dist_bitsets = []
        
        for (px, py) in candidates:
            # Check if we can add (px, py)
            valid = True
            seen_dists = 0  # Bitset for distances from P to points already in S
            new_dists = []   # List of squared distances for updating existing points
            
            for i, (ax, ay) in enumerate(points):
                d2 = (px - ax) ** 2 + (py - ay) ** 2
                new_dists.append(d2)
                
                # Check 1: P is not the apex
                # If this distance was already seen for another point A', then d(P, A) == d(P, A')
                if (seen_dists >> d2) & 1:
                    valid = False
                    break
                
                # Check 2 & 3: Existing points are not the apex
                # If this distance exists in A's neighbor distances, then d(P, A) == d(A, B) for some B
                # This makes A the apex of triangle P-A-B.
                # Symmetrically, when we process B, we will catch d(P, B) == d(B, A).
                if (dist_bitsets[i] >> d2) & 1:
                    valid = False
                    break
                
                # Mark this distance as seen for P
                seen_dists |= (1 << d2)
            
            if not valid:
                continue
            
            # All checks passed, add the point
            points.append((px, py))
            
            # Update dist_bitsets
            # For the new point, its dist_bitset is the set of all distances to previous points
            dist_bitsets.append(seen_dists)
            
            # For each existing point i, add the new distance d2 to its bitset
            for i, d2 in enumerate(new_dists):
                dist_bitsets[i] |= (1 << d2)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points.copy()
            
    return best_points
# EVOLVE-BLOCK-END
