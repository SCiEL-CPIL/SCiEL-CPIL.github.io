import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    D = 11
    
    # Generate all weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    vec_support = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)
    
    n_total = len(all_vectors)
    
    # Precompute incompatibility masks
    incompat_mask = [0] * n_total
    
    # Build tri_to_supports
    tri_to_supports = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_supports[tri].append(i)
            
    # Group vectors by support
    vectors_by_support = [[] for _ in range(n_supports)]
    for idx in range(n_total):
        vectors_by_support[vec_support[idx]].append(idx)
        
    # Store vectors as tuples for faster access
    vec_tuples = [tuple(v) for v in all_vectors]
    
    # For each vector, determine conflicts
    for s_idx in range(n_supports):
        S = supports[s_idx]
        vecs_in_S = vectors_by_support[s_idx]
        
        for v_idx in vecs_in_S:
            v = vec_tuples[v_idx]
            
            # Get 3-subsets of S
            for tri in itertools.combinations(S, 3):
                # Get all supports sharing this tri
                for s_prime in tri_to_supports[tri]:
                    if s_prime == s_idx:
                        continue
                    
                    # Check vectors in s_prime
                    for w_idx in vectors_by_support[s_prime]:
                        w = vec_tuples[w_idx]
                        
                        # Check if signs on tri are same
                        match = True
                        for pos in tri:
                            if v[pos] != w[pos]:
                                match = False
                                break
                            
                        if match:
                            # Conflict!
                            incompat_mask[v_idx] |= (1 << w_idx)
                            incompat_mask[w_idx] |= (1 << v_idx)
                            
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected_mask = 0
    
    max_restarts = 5000 # High number of restarts possible due to speed
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 2:
            break
        
        # Greedy Phase
        order = rng.permutation(n_total)
        
        selected_mask = 0
        selected_count = 0
        
        for idx in order:
            if (incompat_mask[idx] & selected_mask) == 0:
                selected_mask |= (1 << idx)
                selected_count += 1
        
        # Local Search Phase (Fill)
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 1:
                break
            
            for idx in range(n_total):
                if (selected_mask >> idx) & 1:
                    continue
                
                if (incompat_mask[idx] & selected_mask) == 0:
                    selected_mask |= (1 << idx)
                    selected_count += 1
                    improved = True
                    
        if selected_count > best_k:
            best_k = selected_count
            best_selected_mask = selected_mask
            
    if best_k == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Extract selected vectors
    selected_indices = []
    m = best_selected_mask
    idx = 0
    while m:
        if m & 1:
            selected_indices.append(idx)
        m >>= 1
        idx += 1
        
    result = np.array([all_vectors[i] for i in selected_indices], dtype=np.int64)
    
    return result
