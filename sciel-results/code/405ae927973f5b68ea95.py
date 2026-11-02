import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Generate candidate vectors
    # We want integer vectors v in Z^11 such that:
    # min_{i != j} ||v_i - v_j|| >= max_i ||v_i||
    # Equivalently, after normalizing to unit norm, min pairwise angular distance condition.
    # 
    # Key insight: If all vectors have the same norm (e.g., norm squared = 2), then the condition becomes:
    # ||v_i - v_j||^2 >= 2 for all i != j
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 - 2*v_i.v_j
    # So we need 4 - 2*v_i.v_j >= 2 => v_i.v_j <= 1
    #
    # For norm squared = 2 vectors (two non-zero entries of +/-1):
    # v_i.v_j can be -2, -1, 0, 1, 2
    # We need v_i.v_j <= 1, so we just need to exclude pairs with dot product 2 (same vector) 
    # and dot product -2 (opposite vector).
    # Actually wait: if v_i.v_j = 2, that means v_i = v_j (same vector), which we exclude by uniqueness.
    # If v_i.v_j = -2, that means v_j = -v_i, and ||v_i - v_j||^2 = ||2v_i||^2 = 8 >= 2, so that's fine.
    # 
    # So for norm-squared-2 vectors, ANY set of distinct vectors works! 
    # The number of such vectors is C(11,2) * 4 = 55 * 4 = 220.
    # 
    # But we can do better by including vectors of higher norm.
    # If we mix norms, the constraint is: ||v_i - v_j||^2 >= max(||v_i||^2, ||v_j||^2, max_norm^2)
    # Actually the condition is: min pairwise distance >= max norm.
    # Let M = max_i ||v_i||. Then we need ||v_i - v_j|| >= M for all i != j.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= M^2
    #
    # Strategy: Use a pool of vectors with norm squared in {1, 2, 3, 4, 5} and greedily select.
    # But the greedy approach from the seed was slow. Let's use a smarter approach.
    #
    # Better strategy: 
    # 1. Start with all norm-squared-2 vectors (220 vectors) - this is a valid configuration.
    # 2. Try to add vectors of higher norm by checking compatibility.
    # 3. Use local search / simulated annealing to improve.
    
    # Generate all norm-squared-2 vectors
    vecs = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vecs.append(v)
    
    # Also generate norm-squared-1 vectors (unit vectors with one +/-1)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            vecs.append(v)
    
    # Generate norm-squared-3 vectors (one entry +/-1, two entries +/-1 with 1+1+1=3)
    # Actually norm squared 3: entries with squares summing to 3: (1,1,1) pattern
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vecs.append(v)
    
    # Generate norm-squared-4 vectors: (2,0,...) or (1,1,1,1) or (2,1,1) patterns
    # (2,0,...): one entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vecs.append(v)
    # (1,1,1,1): four entries of +/-1
    for idxs in itertools.combinations(range(D), 4):
        for vals in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, val in zip(idxs, vals):
                v[idx] = val
            vecs.append(v)
    # (2,1,1): one entry +/-2, two entries +/-1
    for i, j, k in itertools.combinations(range(D), 3):
        for s2 in (1, -1):
            for s1 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i] = 2 * s2
                    v[j] = s1
                    v[k] = s3
                    vecs.append(v)
    
    # Generate norm-squared-5 vectors: (2,1,1,1) or (1,1,1,1,1)
    # (2,1,1,1): one entry +/-2, three entries +/-1
    for idxs in itertools.combinations(range(D), 4):
        for pos2 in range(4):
            for s2 in (1, -1):
                for vals in itertools.product((1, -1), repeat=3):
                    v = [0] * D
                    idxs_list = list(idxs)
                    v[idxs_list[pos2]] = 2 * s2
                    other_idx = 0
                    for idx in idxs_list:
                        if idx != idxs_list[pos2]:
                            v[idx] = vals[other_idx]
                            other_idx += 1
                    vecs.append(v)
    # (1,1,1,1,1): five entries of +/-1
    for idxs in itertools.combinations(range(D), 5):
        for vals in itertools.product((1, -1), repeat=5):
            v = [0] * D
            for idx, val in zip(idxs, vals):
                v[idx] = val
            vecs.append(v)
    
    # Convert to numpy arrays and deduplicate
    vec_set = set()
    for v in vecs:
        vec_set.add(tuple(v))
    vecs = [np.array(v, dtype=np.int64) for v in vec_set]
    n_vecs = len(vecs)
    
    # Precompute norms squared
    norms_sq = np.array([np.dot(v, v) for v in vecs], dtype=np.int64)
    max_norm_sq = np.max(norms_sq)
    
    # Precompute pairwise distances squared (this is O(n^2 * D) which might be large)
    # n_vecs could be large, so let's be careful
    # Actually, let's estimate: 
    # norm 1: 22
    # norm 2: 220
    # norm 3: C(11,3)*8 = 165*8 = 1320
    # norm 4: 22 + C(11,4)*16 + C(11,3)*8 = 22 + 330*16 + 165*8 = 22 + 5280 + 1320 = 6622
    # norm 5: C(11,4)*4*8 + C(11,5)*32 = 330*32 + 462*32 = 10560 + 14784 = 25344
    # Total ~33k vectors. Pairwise distances would be ~33k^2 * 11 ~ 12 billion ops, too slow.
    # 
    # Let's use a different approach: greedy with efficient distance checking.
    
    # Greedy approach with multiple restarts
    best_set = []
    best_size = 0
    best_max_norm_sq = 0
    
    # First, get a baseline from just norm-2 vectors
    norm2_vecs = [v for v, n2 in zip(vecs, norms_sq) if n2 == 2]
    if len(norm2_vecs) > best_size:
        best_size = len(norm2_vecs)
        best_set = norm2_vecs[:]
        best_max_norm_sq = 2
    
    # Now try to build a larger set using all vectors
    # Use a greedy approach with random ordering
    
    num_restarts = 30
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random permutation
        perm = rng.permutation(n_vecs)
        current_set = []
        current_max_norm_sq = 0
        
        for idx in perm:
            v = vecs[idx]
            v_norm_sq = norms_sq[idx]
            new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
            
            # Check if v is compatible with all in current_set
            # Condition: ||v - u||^2 >= new_max_norm_sq for all u in current_set
            compatible = True
            for u in current_set:
                diff = v - u
                dist_sq = np.dot(diff, diff)
                if dist_sq < new_max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                current_set.append(v)
                current_max_norm_sq = new_max_norm_sq
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            best_max_norm_sq = current_max_norm_sq
            
            # Try local improvement: add more vectors if possible
            if time.time() < deadline - 5:
                added = True
                while added and time.time() < deadline - 2:
                    added = False
                    for idx in range(n_vecs):
                        if time.time() > deadline - 2:
                            break
                        v = vecs[idx]
                        # Check if v is already in current_set
                        if any(np.array_equal(v, u) for u in current_set):
                            continue
                        v_norm_sq = norms_sq[idx]
                        new_max_norm_sq = max(best_max_norm_sq, v_norm_sq)
                        compatible = True
                        for u in current_set:
                            diff = v - u
                            dist_sq = np.dot(diff, diff)
                            if dist_sq < new_max_norm_sq:
                                compatible = False
                                break
                        if compatible:
                            current_set.append(v)
                            best_max_norm_sq = new_max_norm_sq
                            best_size += 1
                            added = True
    
    return np.array(best_set, dtype=np.int64)
