import numpy as np
import random
import time

N = 64
M = N * N
DEADLINE_MARGIN = 290

# Pre-compute coordinates
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

# Pre-compute squared distances using broadcasting
coords = points[:, np.newaxis, :] - points[np.newaxis, :, :]
D = np.sum(coords * coords, axis=-1).astype(np.int16)

def construct():
    start_time = time.time()
    best_set = []
    rng = random.Random()
    
    # Run multiple restarts until deadline
    while time.time() - start_time < DEADLINE_MARGIN:
        candidates = list(range(M))
        rng.shuffle(candidates)
        
        S = []
        # active_dist_sets[i] holds the set of squared distances from S[i] to all other points in S
        active_dist_sets = []
        
        for p in candidates:
            if not S:
                S.append(p)
                active_dist_sets.append(set())
                continue
            
            # 1. Check distances from p to all points in S are distinct
            dists_to_S = D[p, S]
            
            # Check for duplicates
            if len(dists_to_S) != len(set(dists_to_S)):
                continue
            
            # 2. Check if adding p creates an isosceles triangle with apex q for any q in S
            valid = True
            for i in range(len(S)):
                q = S[i]
                if D[p, q] in active_dist_sets[i]:
                    valid = False
                    break
            
            if valid:
                S.append(p)
                active_dist_sets.append(set(dists_to_S))
                
                # Update the distance sets for all existing points in S
                for i in range(len(S) - 1):
                    q = S[i]
                    active_dist_sets[i].add(D[p, q])
                    
        if len(S) > len(best_set):
            best_set = S
            
    # Convert indices back to coordinates
    return [(points[p, 0], points[p, 1]) for p in best_set]
