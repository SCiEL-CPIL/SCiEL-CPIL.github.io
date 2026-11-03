import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate candidates with norm squared <= 3
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    
    candidates = np.array(candidates)
    N = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # We want max norm squared R2 = 3
    # Condition: ||u - v||^2 >= 3
    
    best_set = []
    
    # Strategy: Greedy with random restarts
    # To speed up, we can compute the conflict matrix or just check pairwise.
    # N is ~1500. N^2 is 2.25M. Checking all pairs is fast.
    # But we do this in a loop.
    
    # Let's compute the degree of each node in the conflict graph to sort them.
    # Conflict if dist_sq < 3.
    
    # We will run a few iterations with different orderings.
    
    for run in range(50): # Max 50 restarts
        if time.time() > deadline:
            break
            
        # Random shuffle
        order = np.random.permutation(N)
        
        selected = []
        selected_indices = set()
        
        # To make it faster, we can maintain a list of available candidates
        # But checking against selected is O(K).
        
        for idx in order:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            is_valid = True
            for sel_idx in selected:
                # Check distance
                diff = v - candidates[sel_idx]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 3:
                    is_valid = False
                    break
            
            if is_valid:
                selected.append(idx)
        
        if len(selected) > len(best_set):
            best_set = selected
            
        # Early stop if we found a very good solution?
        # Kissing number 504. If we hit 400+, maybe stop?
        # But let's run a few more.
        
    # Convert best_set to array
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set]
