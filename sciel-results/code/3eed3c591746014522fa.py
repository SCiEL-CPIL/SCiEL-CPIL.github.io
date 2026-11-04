import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Generate Type A: Two 2s
    # Vectors like (2, 2, 0, ...), (2, -2, 0, ...), etc.
    type_A = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                type_A.append(v)
    
    S = np.array(type_A, dtype=np.int64)
    
    # 2. Generate Candidates: Type B (One 2, four 1s) and Type C (Eight 1s)
    candidates = []
    
    # Type B
    for i in range(D):
        for si in (1, -1):
            # Choose 4 indices from the remaining 10
            other_indices = [k for k in range(D) if k != i]
            for comb in itertools.combinations(other_indices, 4):
                for signs in itertools.product((1, -1), repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * si
                    for idx, s in zip(comb, signs):
                        v[idx] = s
                    candidates.append(v)
                    
    if time.time() > deadline:
        return S

    # Type C
    for comb in itertools.combinations(range(D), 8):
        for signs in itertools.product((1, -1), repeat=8):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    if time.time() > deadline:
        return S

    candidates = np.array(candidates, dtype=np.int64)
    
    # Shuffle candidates to avoid bias?
    # np.random.shuffle(candidates)
    
    # 3. Greedy Add
    # We need to check if candidate v is compatible with all u in S.
    # Condition: u . v <= 4
    
    # To speed up, we can check in batches or just loop.
    # Looping 116k times is fine.
    
    for v in candidates:
        if time.time() > deadline:
            break
        
        # Check compatibility
        # dots = S @ v
        # if np.max(dots) <= 4:
        #     S = np.vstack((S, v[None, :]))
        
        # Optimization: 
        # If S is large, S @ v is O(K*D).
        # We can also check if v is compatible with v itself? 
        # v . v = 8. 8 <= 4 is False.
        # Wait, the condition is for PAIRWISE distance.
        # u, v in S.
        # If we add v, we check against existing S.
        # We do NOT check v against itself (distance 0).
        # But we must ensure v is not a duplicate?
        # If v is in S, dist 0. 
        # But we are generating unique vectors.
        
        dots = S @ v
        if np.max(dots) <= 4:
            S = np.vstack((S, v[None, :]))
            
    return S
