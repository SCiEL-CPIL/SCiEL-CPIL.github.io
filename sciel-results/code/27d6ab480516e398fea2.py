import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 50s margin
    
    # Flatten grid to indices
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances between all pairs of points
    # dist2[i][j] is squared distance between points[i] and points[j]
    # Using numpy for speed
    coords = np.array(points)
    # Compute pairwise squared distances
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # dist2 = np.sum(diff**2, axis=2)
    # This creates a 4096x4096 array, which is fine.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    
    best_set = []
    best_size = 0
    
    # Greedy with random restarts
    indices = list(range(num_points))
    
    while time.time() < deadline:
        random.shuffle(indices)
        current_set = []
        
        for idx in indices:
            # Check if idx can be added to current_set
            valid = True
            # If current_set is large, this loop is expensive
            # We can break early if invalid
            for i in range(len(current_set)):
                ia = current_set[i]
                for j in range(i+1, len(current_set)):
                    ib = current_set[j]
                    d_pa = dist2[idx][ia]
                    d_pb = dist2[idx][ib]
                    d_ab = dist2[ia][ib]
                    if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(idx)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # Optional: Local search? 
            # For now, just restarts.
            
    return [points[i] for i in best_set]
