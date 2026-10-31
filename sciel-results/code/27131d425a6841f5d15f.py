import itertools
import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate W1 vectors (norm sq 4)
    w1_vectors = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            w1_vectors.append(tuple(v))
    
    # 2. Generate W4 vectors (norm sq 4)
    w4_vectors = []
    w4_indices = []
    idx = 0
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, i in enumerate(support):
                v[i] = signs[k]
            vec = tuple(v)
            w4_vectors.append(vec)
            w4_indices.append(idx)
            idx += 1
            
    n_w4 = len(w4_vectors)
    
    # 3. Precompute incompatibility graph for W4
    # Incompatible if: share 3 indices with same signs.
    # Map vector to index
    vec_to_idx = {v: i for i, v in enumerate(w4_vectors)}
    
    incompat = [set() for _ in range(n_w4)]
    
    # To speed up finding incompatible pairs:
    # For each vector u, find v that share 3 indices with same signs.
    # This is O(N * 4 * 7 * 2) roughly.
    # 4 choices of which index to drop from support.
    # 7 choices for the new index.
    # 2 choices for the sign of the new index.
    
    for u_idx, u in enumerate(w4_vectors):
        support_u = [i for i in range(D) if u[i] != 0]
        # For each of the 4 indices in support_u, try to form a triple
        for drop_idx in range(4):
            triple = [support_u[j] for j in range(4) if j != drop_idx]
            # Signs on the triple
            triple_signs = [u[i] for i in triple]
            
            # The 4th index of v can be anything not in triple
            # But v must have size 4.
            # So v's support is triple + {new_idx}.
            # new_idx can be support_u[drop_idx] (same support) or outside.
            
            # Case 1: new_idx = support_u[drop_idx]
            # Then v has same support as u.
            # v must agree on triple.
            # v's 4th sign can be u[drop_idx] or -u[drop_idx].
            # If v[drop_idx] == u[drop_idx], v == u. (Skip)
            # If v[drop_idx] == -u[drop_idx], v != u.
            # Check compatibility: m=1 (one diff sign). Compatible.
            # So no incompatibility here.
            
            # Case 2: new_idx not in support_u
            # There are 11 - 4 = 7 such indices.
            for new_idx in range(D):
                if new_idx in support_u:
                    continue
                for new_sign in (1, -1):
                    # Construct v
                    v = list(u)
                    v[support_u[drop_idx]] = 0
                    v[new_idx] = new_sign
                    v_tuple = tuple(v)
                    # v agrees on triple by construction?
                    # We didn't change triple indices.
                    # So v agrees on triple.
                    # Thus v is incompatible with u.
                    v_idx = vec_to_idx.get(v_tuple)
                    if v_idx is not None:
                        incompat[u_idx].add(v_idx)
                        incompat[v_idx].add(u_idx)
                        
    # 4. Greedy MIS with restarts
    best_set_size = 0
    best_indices = []
    
    # Precompute neighbors list for faster access
    neighbors = [list(s) for s in incompat]
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        order = list(range(n_w4))
        random.shuffle(order)
        
        in_set = [False] * n_w4
        conflict_count = [0] * n_w4
        current_size = 0
        
        for u_idx in order:
            if conflict_count[u_idx] == 0:
                in_set[u_idx] = True
                current_size += 1
                for v_idx in neighbors[u_idx]:
                    if not in_set[v_idx]:
                        conflict_count[v_idx] += 1
                        
        if current_size > best_set_size:
            best_set_size = current_size
            best_indices = [i for i in range(n_w4) if in_set[i]]
            
    # 5. Construct result
    result_vectors = []
    for i in best_indices:
        result_vectors.append(list(w4_vectors[i]))
    for v in w1_vectors:
        result_vectors.append(list(v))
        
    return np.array(result_vectors, dtype=np.int64)
