import time
import random
import math

N = 64
# Precompute squared distances
# dist2[i][j] where i, j are indices in 0..N*N-1
# i = x1*N + y1, j = x2*N + y2
# This matrix is 4096x4096, which is 16M entries.
# int16 or int32? Max dist squared is (63^2 + 63^2) = 7938. Fits in int16.
import numpy as np

# Initialize distance matrix
# We only need upper triangle or full matrix for fast access.
# Full matrix is easier.
# Memory: 4096 * 4096 * 2 bytes (int16) = 32 MB. OK.

# But constructing it in Python loop is slow.
# Let's do it with numpy.
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

# dist2[i, j] = (xs[i]-xs[j])^2 + (ys[i]-ys[j])^2
# This creates a 4096x4096 array.
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx*dx + dy*dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    all_indices = list(range(N*N))
    best_set = []
    best_size = 0
    
    # We will try several random permutations
    # A greedy algorithm with random restarts
    # To speed up, we can use a fixed list and shuffle a copy, or use random.sample
    # But shuffling 4096 items is fast.
    
    # Optimization: 
    # Instead of checking all 4096 points, we can stop early if we find a set larger than best_size?
    # No, we need to check all points to see if they can be added.
    
    # Let's try 500 attempts first to see timing.
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_indices)
        current_set = []
        # used_dists[i] is a set of squared distances from current_set[i] to other points in current_set
        # We store indices of current_set in a list `set_indices`
        set_indices = []
        used_dists = [] # list of sets
        
        for p_idx in all_indices:
            # Check if p_idx can be added
            valid = True
            if not set_indices:
                valid = True
            else:
                # 1. Distances from p to all existing points must be distinct
                # Extract distances
                # dist2[p_idx, q_idx] for q_idx in set_indices
                # We can do this with numpy slicing if set_indices is an array, but it's a list.
                # Manual loop is probably faster for small k.
                dists = []
                for q_idx in set_indices:
                    dists.append(dist2[p_idx, q_idx])
                
                if len(set(dists)) != len(dists):
                    valid = False
            
            if valid:
                # 2. For each existing point q, distance from q to p must not be in used_dists[q]
                for i, q_idx in enumerate(set_indices):
                    d_sq = dist2[p_idx, q_idx]
                    if d_sq in used_dists[i]:
                        valid = False
                        break
            
            if valid:
                # Add p_idx to current_set
                set_indices.append(p_idx)
                new_dist_set = set()
                for q_idx in set_indices[:-1]:
                    new_dist_set.add(dist2[p_idx, q_idx])
                used_dists.append(new_dist_set)
                
                # Update used_dists for existing points
                for i, q_idx in enumerate(set_indices[:-1]):
                    used_dists[i].add(dist2[p_idx, q_idx])
        
        if len(set_indices) > best_size:
            best_size = len(set_indices)
            # Convert indices back to (x, y)
            best_set = [(idx // N, idx % N) for idx in set_indices]
            
            # Optional: if we reach a high size, we might want to try local search?
            # For now, just keep the best.
            
    return best_set
