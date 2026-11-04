import numpy as np
import itertools
import time

D = 11

def construct():
    # 1. Generate all vectors with exactly 4 non-zero entries (+/- 1)
    all_vectors = []
    vec_index = {} # (support_tuple, signs_tuple) -> index
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append(v)
            vec_index[(support, signs)] = idx
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph using local neighborhood generation
    # adj[i] is a set of indices j such that i and j conflict
    adj = [set() for _ in range(N)]
    
    # For each vector, find its conflicts
    # v has support S (len 4) and signs s (len 4)
    for i, v in enumerate(all_vectors):
        # Reconstruct support and signs
        # This is inefficient. Better to store them.
        pass
        
    # Let's store support and signs during generation
    all_vectors_info = [] # (v, support, signs)
    vec_index = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors_info.append((v, support, signs))
            vec_index[(support, signs)] = idx
            idx += 1
            
    N = len(all_vectors_info)
    adj = [set() for _ in range(N)]
    
    for i, (v, support, signs) in enumerate(all_vectors_info):
        # Generate conflicts
        # Subsets of support of size 3
        for j in range(4):
            # Remove support[j]
            shared_support = tuple(support[k] for k in range(4) if k != j)
            shared_signs = tuple(signs[k] for k in range(4) if k != j)
            
            # The 4th index in the conflict vector must be NOT in support
            for new_idx in range(D):
                if new_idx in support:
                    continue
                # The sign at new_idx can be 1 or -1
                for new_sign in [1, -1]:
                    # Construct the full support and signs for the potential conflict
                    # The conflict vector has support shared_support + (new_idx,)
                    # And signs shared_signs + (new_sign,)
                    # We need to sort the support and align signs to use as key
                    
                    new_support_list = list(shared_support) + [new_idx]
                    new_signs_list = list(shared_signs) + [new_sign]
                    
                    # Sort by index to normalize
                    pairs = sorted(zip(new_support_list, new_signs_list))
                    norm_support = tuple(p[0] for p in pairs)
                    norm_signs = tuple(p[1] for p in pairs)
                    
                    key = (norm_support, norm_signs)
                    if key in vec_index:
                        j_idx = vec_index[key]
                        if j_idx != i:
                            adj[i].add(j_idx)
                            adj[j_idx].add(i)

    # 3. Solve MIS
    # Convert adj to bitmasks for faster operations in solver
    adj_masks = [0] * N
    for i in range(N):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_masks[i] = mask
        
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return x.bit_count()
        
    def greedy_mis(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            # Find vertex with min degree in remaining
            min_deg = float('inf')
            v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = popcount(adj_masks[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
                temp ^= lsb
                
            if v == -1:
                break
                
            selected |= (1 << v)
            remaining &= ~(adj_masks[v] | (1 << v))
            
        return selected, popcount(selected)

    start_time = time.time()
    time_limit = 115.0

    def solve(candidates, current_mask, current_count):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_count > best_size:
                best_size = current_count
                best_mask = current_mask
            return
            
        if current_count + popcount(candidates) <= best_size:
            return
            
        g_mask, g_count = greedy_mis(candidates)
        if current_count + g_count > best_size:
            best_size = current_count + g_count
            best_mask = current_mask | g_mask
            
        if current_count + popcount(candidates) <= best_size:
            return
            
        # Select vertex with max degree
        max_deg = -1
        v = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = popcount(adj_masks[u] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = u
            temp ^= lsb
            
        if v == -1:
            return
            
        v_bit = 1 << v
        
        new_candidates_1 = candidates & ~(adj_masks[v] | v_bit)
        solve(new_candidates_1, current_mask | v_bit, current_count + 1)
        
        new_candidates_2 = candidates & ~v_bit
        solve(new_candidates_2, current_mask, current_count)

    initial_candidates = (1 << N) - 1
    solve(initial_candidates, 0, 0)
    
    if best_size == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        i = lsb.bit_length() - 1
        result_vectors.append(all_vectors_info[i][0])
        temp ^= lsb
        
    return np.array(result_vectors, dtype=np.int64)
