import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N
# Precompute coordinates
ys, xs = np.meshgrid(np.arange(N), np.arange(N))
coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.int32)

# Precompute squared distances
# dist_sq[i, j] = squared distance between point i and point j
# This is a 4096x4096 matrix.
# 4096^2 * 8 bytes = 134 MB. Acceptable.
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diff**2, axis=2).astype(np.int32)
del diff

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    best_set = []
    best_size = 0
    
    # We will run multiple greedy restarts
    # Each restart shuffles the points and greedily adds them
    
    # To speed up, we can pre-shuffle indices once per restart
    indices = np.arange(NUM_POINTS)
    
    # We can try to bias the order? 
    # Random is a good baseline.
    
    while time.time() < deadline:
        random.shuffle(indices)
        
        S = []
        # dist_sets[i] will store the set of squared distances from i to all points in S
        # We only care about i in S, but using a full array is easier.
        # Actually, we can just use a list of sets for points in S?
        # No, we need to check `d in dist_sets[A]` for A in S.
        # So we need dist_sets[A] to be up to date.
        # Let's use a list of sets, indexed by point ID.
        dist_sets = [set() for _ in range(NUM_POINTS)]
        
        # Optimization: Instead of clearing 4096 sets every restart, 
        # we can just track which sets are non-empty and clear them at the end?
        # Or just create new ones. 4096 sets creation is fast.
        
        for P in indices:
            # Check if P can be added
            dists = set()
            valid = True
            
            # Iterate through current S
            for A in S:
                d = dist_sq[P, A]
                if d in dists:
                    valid = False
                    break
                dists.add(d)
                
                # Check if d is already a distance from A to some point in S
                if d in dist_sets[A]:
                    valid = False
                    break
            
            if valid:
                S.append(P)
                # Update dist_sets
                # Add distance P-A to dist_sets[A] for all A in S
                # And populate dist_sets[P] with distances to all A in S
                for A in S:
                    d = dist_sq[P, A]
                    dist_sets[A].add(d)
                    dist_sets[P].add(d)
                
                # Optional: Early exit if we have enough time? No, keep going.
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we have very little time left, break
        if time.time() > deadline - 5:
            break
            
    return [(int(coords[p][0]), int(coords[p][1])) for p in best_set]
