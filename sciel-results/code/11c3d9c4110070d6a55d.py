import numpy as np
import time
import random

N = 64
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
pts_arr = np.array(POINTS, dtype=np.int32)
# dist_sq[i, j] is squared distance between POINTS[i] and POINTS[j]
# Using broadcasting
diff = pts_arr[:, None, :] - pts_arr[None, :, :]
dist_sq = np.sum(diff * diff, axis=2)

def construct():
    best_set = []
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin for return and safety
    
    indices = list(range(NUM_POINTS))
    
    while time.time() < deadline:
        random.shuffle(indices)
        current_set = []
        # dist_sets maps point index to a set of squared distances to other points in current_set
        dist_sets = {}
        
        for p_idx in indices:
            valid = True
            new_distances = []
            
            # Check against all points in current_set
            for q_idx in current_set:
                d = dist_sq[q_idx, p_idx]
                if d in dist_sets[q_idx]:
                    valid = False
                    break
                new_distances.append(d)
            
            if valid:
                current_set.append(p_idx)
                dist_sets[p_idx] = set(new_distances)
                
                # Update existing points
                for q_idx in current_set[:-1]:
                    dist_sets[q_idx].add(dist_sq[q_idx, p_idx])
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    return [POINTS[i] for i in best_set]
