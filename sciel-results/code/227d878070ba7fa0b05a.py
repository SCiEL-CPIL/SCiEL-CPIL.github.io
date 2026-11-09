import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 15s margin
    
    # Generate all V3 vectors: 3 non-zero entries, each +/-1
    # There are C(11,3) * 8 = 1320 such vectors
    indices = range(D)
    v3_candidates = []
    for combo in itertools.combinations(indices, 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for k, s in zip(combo, signs):
                v[k] = s
            v3_candidates.append(v)
    
    v3_candidates = np.array(v3_candidates)
    
    # We want the largest subset with pairwise dot product <= 1
    # Greedy with random restarts
    best_set = []
    n_cands = len(v3_candidates)
    
    # Pre-compute dot products? 1320x1320 is ~1.7M entries. 
    # 1.7M * 8 bytes = 14MB. Acceptable.
    # But we can do it on the fly or in batches.
    # Let's pre-compute to speed up the greedy.
    dots = v3_candidates @ v3_candidates.T
    
    # Mask of valid pairs (dot <= 1)
    # Note: diagonal is 3. We ignore diagonal.
    valid_mask = dots <= 1
    
    # We need to find a large clique in the graph defined by valid_mask?
    # No, we need an independent set in the graph of INVALID pairs (dot > 1).
    # Or a clique in the graph of VALID pairs.
    # Greedy: pick a vertex, remove all its invalid neighbors.
    
    def greedy_select(mask, order):
        selected = []
        available = np.ones(len(order), dtype=bool)
        for idx in order:
            if not available[idx]:
                continue
            # Check if this vertex is compatible with all already selected?
            # In the standard greedy for independent set on conflict graph:
            # We pick a node, and remove all nodes that conflict with it.
            # Here, "conflict" means dot > 1.
            # So we remove all j such that dots[idx, j] > 1.
            
            # Find conflicts
            conflicts = dots[idx] > 1
            # Mark conflicts as unavailable
            available[conflicts] = False
            available[idx] = False # mark as used/unavailable for future picks
            selected.append(idx)
            
        return selected

    best_count = 0
    best_indices = []
    
    for _ in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        # Do NOT shuffle v3_candidates, as dots is precomputed for the original order.
        # Only shuffle the processing order.
        order = list(range(n_cands))
        random.shuffle(order)
        
        selected = greedy_select(valid_mask, order)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = selected
            
        # If we already beat the known best (348), we can stop early or continue a bit
        if best_count > 400:
             break

    if best_count == 0:
        # Fallback to the 220 seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return v3_candidates[best_indices]
