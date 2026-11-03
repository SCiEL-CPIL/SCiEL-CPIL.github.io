import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0 # 150s limit, leave margin
    
    # Generate all weight-4 vectors
    # Indices 0..10
    supports = list(itertools.combinations(range(D), 4))
    vectors = []
    vec_to_idx = {}
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(s, signs):
                v[idx] = val
            # Normalize to tuple for hashing
            key = tuple(v)
            if key not in vec_to_idx:
                vec_to_idx[key] = len(vectors)
                vectors.append(v)
    
    n_vecs = len(vectors)
    
    # Precompute dot products? Too big (5280^2).
    # Compute on the fly or use bitsets for conflicts.
    # Conflict: dot > 2.
    
    # Greedy Independent Set
    best_set = []
    best_size = 0
    
    # Random restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random order
        indices = list(range(n_vecs))
        random.shuffle(indices)
        
        current_set = []
        current_set_set = set()
        
        # Greedy add
        for idx in indices:
            v = vectors[idx]
            conflict = False
            # Check against current_set
            # Optimization: only check if necessary?
            # For small sets, linear scan is fine.
            for u_idx in current_set_set:
                u = vectors[u_idx]
                # Dot product
                dot = 0
                for i in range(D):
                    if v[i] != 0 and u[i] != 0:
                        dot += v[i] * u[i]
                if dot > 2:
                    conflict = True
                    break
            
            if not conflict:
                current_set.append(idx)
                current_set_set.add(idx)
                
        # Local Search: try to improve
        # This is expensive. Maybe just keep the best greedy result.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # Try to add antipodes
    final_vectors = [vectors[i] for i in best_set]
    final_set_set = set(tuple(v) for v in final_vectors)
    
    for i in best_set:
        v = vectors[i]
        neg_v = [-x for x in v]
        key = tuple(neg_v)
        if key in final_set_set:
            continue
        # Check conflict with existing
        conflict = False
        for u in final_vectors:
            dot = 0
            for j in range(D):
                if neg_v[j] != 0 and u[j] != 0:
                    dot += neg_v[j] * u[j]
            if dot > 2:
                conflict = True
                break
        if not conflict:
            final_vectors.append(neg_v)
            final_set_set.add(key)
            
    if not final_vectors:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(final_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
