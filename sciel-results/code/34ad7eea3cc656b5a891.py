import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate all candidate points
    points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    # Use multiple random restarts to find a larger set
    # The order of adding points matters for greedy algorithms
    num_restarts = 100
    
    for _ in range(num_restarts):
        random.shuffle(points)
        S = []
        used_dists = set()
        
        for p in points:
            px, py = p
            new_dists = set()
            valid = True
            
            # Check distances to all existing points in S
            for q in S:
                qx, qy = q
                d = (px - qx)**2 + (py - qy)**2
                
                # If this distance was already used by any pair in S, reject
                if d in used_dists:
                    valid = False
                    break
                # If this distance is already used by another point in S with P, reject
                if d in new_dists:
                    valid = False
                    break
                
                new_dists.add(d)
            
            if valid:
                S.append(p)
                used_dists.update(new_dists)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # Early stopping if we reach a theoretical upper bound or high value
        # Max possible distinct distances in [0,64)^2 is 2*63^2 = 7938
        # Max k such that k(k-1)/2 <= 7938 is ~126
        # If we get close to 120, we can stop
        if best_size > 110:
            break

    return best_set
# EVOLVE-BLOCK-END
