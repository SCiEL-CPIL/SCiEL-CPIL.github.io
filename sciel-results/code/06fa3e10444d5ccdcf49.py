import numpy as np
import itertools
import time

D = 11

def construct():
    # 1. Generate all vectors with exactly 4 non-zero entries (+/- 1)
    all_vectors = []
    support_map = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        support_map[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph using bitmasks
    # adj[i] is a bitmask of nodes conflicting with i
    adj = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Precompute sign signatures for faster comparison?
    # For each support, map sign tuple to list of vector indices
    # Actually, we just need to check conflicts between pairs of supports with intersection 3.
    
    # Optimization: Group vectors by support.
    # For each pair of supports s1, s2 with intersection size 3:
    #   For each v1 in s1, v2 in s2:
    #     Check if signs match on intersection.
    
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                # Get positions in s1 and s2
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                # Map sign signature on intersection to indices for s2
                s2_sig_map = {}
                for idx2 in vecs_s2:
                    signs2 = all_vectors[idx2][2]
                    sig2 = tuple(signs2[p] for p in pos_in_s2)
                    if sig2 not in s2_sig_map:
                        s2_sig_map[sig2] = []
                    s2_sig_map[sig2].append(idx2)
                    
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    sig1 = tuple(signs1[p] for p in pos_in_s1)
                    
                    if sig1 in s2_sig_map:
                        for idx2 in s2_sig_map[sig1]:
                            adj[idx1] |= (1 << idx2)
                            adj[idx2] |= (1 << idx1)

    # 3. CSP Backtracking Search
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return x.bit_count()

    start_time = time.time()
    time_limit = 115.0

    # Heuristic: Pick vertex with max degree in current available set
    def get_max_degree_vertex(available):
        if not available:
            return -1
        max_deg = -1
        best_v = -1
        temp = available
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = popcount(adj[u] & available)
            if deg > max_deg:
                max_deg = deg
                best_v = u
            temp ^= lsb
        return best_v

    def solve(available, current_mask, current_count):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > time_limit:
            return
            
        if not available:
            if current_count > best_size:
                best_size = current_count
                best_mask = current_mask
            return
            
        # Bound
        avail_count = popcount(available)
        if current_count + avail_count <= best_size:
            return
            
        # Heuristic selection
        v = get_max_degree_vertex(available)
        if v == -1:
            return
            
        v_bit = 1 << v
        
        # Branch 1: Include v
        # Neighbors of v in available
        neighbors = adj[v] & available
        new_available_1 = available & ~(v_bit | neighbors)
        solve(new_available_1, current_mask | v_bit, current_count + 1)
        
        # Branch 2: Exclude v
        new_available_2 = available & ~v_bit
        solve(new_available_2, current_mask, current_count)

    initial_available = (1 << N) - 1
    solve(initial_available, 0, 0)
    
    if best_size == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        i = lsb.bit_length() - 1
        result_vectors.append(all_vectors[i][3])
        temp ^= lsb
        
    return np.array(result_vectors, dtype=np.int64)
