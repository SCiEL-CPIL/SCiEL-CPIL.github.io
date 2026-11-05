import random
import numpy as np
import time

N = 64
MAX_DIST = 2 * (N - 1) ** 2

# EVOLVE-BLOCK-START
def construct():
    # Generate all candidate points
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(points)
    
    # Pre-compute points array for vectorization
    pts_arr = np.array(points)
    
    # Pre-compute conflict scores to guide greedy order
    # Score: number of isosceles triples where this point is the apex
    # For each point p, count duplicates in distances to all other points
    conflict_scores = np.zeros(n_points, dtype=np.int32)
    
    # This might be slow to compute fully for all points if done naively.
    # 4096 points. For each, 4096 distances. 16M ops.
    # We can vectorize the distance matrix calculation?
    # 4096x4096 matrix of squared distances.
    # Memory: 4096*4096*8 bytes = 128 MB. Acceptable.
    
    # Calculate full distance matrix
    # dist_matrix[i, j] = squared distance between points[i] and points[j]
    diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    dist_matrix = np.sum(diff ** 2, axis=2)
    
    # For each row i, count duplicate distances
    # We need sum of binom(count_d, 2) for each distance d in row i
    # This is hard to vectorize directly with unique counts.
    # But we can just compute it for each point in a loop, or use a trick.
    # Actually, for the greedy heuristic, we can just use the number of "close" points or random.
    # Let's skip complex pre-computation for now and rely on random restarts + bitmask speed.
    # If we have time, we can compute a simpler heuristic: 
    # Points in the center have more points at various distances?
    # Actually, points on the edge have fewer neighbors.
    # Let's try sorting by distance from center?
    # Or just random.
    
    best_set = []
    best_size = 0
    
    num_restarts = 500 
    start_time = time.time()
    time_limit = 280
    
    # Pre-allocate bitmask array? No, S changes size.
    # We will store dist_masks as a list of Python ints.
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random permutation
        indices = np.random.permutation(n_points)
        
        S = []
        # dist_masks[i] is a bitmask of distances from S[i] to other points in S
        dist_masks = []
        
        for idx in indices:
            p = pts_arr[idx]
            
            if not S:
                S.append(p)
                dist_masks.append(0)
                continue
            
            # Calculate squared distances from p to all points in S
            # S is a list of arrays, convert to array for vectorization
            S_arr = np.array(S)
            dists = np.sum((S_arr - p) ** 2, axis=1)
            
            # 1. Check for duplicates in new distances (apex at p)
            # Use set for quick check
            if len(set(dists.tolist())) != len(S):
                continue
            
            # 2. Check for isosceles triangles with apex at existing points
            # For each i, check if dists[i] is in dist_masks[i]
            valid = True
            for i in range(len(S)):
                d = dists[i]
                # Check if bit d is set in dist_masks[i]
                if (dist_masks[i] >> int(d)) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Valid point, add to S
            S.append(p)
            new_idx = len(S) - 1
            new_mask = 0
            
            # Update dist_masks
            for i in range(new_idx):
                d = dists[i]
                di = int(d)
                # Set bit d in dist_masks[i]
                dist_masks[i] |= (1 << di)
                # Set bit d in new_mask (distance from new point to S[i])
                new_mask |= (1 << di)
                
            dist_masks.append(new_mask)
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # Early stopping
        if best_size > 50:
             # Try to break if we hit a high bar
             # But let's run a few more to see if we can exceed
             pass

    return best_set
# EVOLVE-BLOCK-END
