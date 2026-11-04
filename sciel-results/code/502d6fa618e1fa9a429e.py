import numpy as np
import itertools
import time
import sys

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
    
    # 2. Build Compatibility Graph (Non-conflict)
    # compat[i] has bit j set if i and j are compatible
    compat = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Precompute conflict info
    # Conflict if:
    # 1. Same support and same signs (identical)
    # 2. Support intersection size 3 and signs match on those 3
    
    # We can build compat by starting with all 1s and removing conflicts
    for i in range(N):
        compat[i] = (1 << N) - 1
        compat[i] &= ~(1 << i) # No self-loop

    # Iterate over pairs of supports to find conflicts
    for i_s, s1 in enumerate(supports):
        vecs_s1 = sup_to_vecs[s1]
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            vecs_s2 = sup_to_vecs[s2]
            
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                # Group s2 vectors by sign signature on intersection
                s2_by_sig = {}
                for idx2 in vecs_s2:
                    signs2 = all_vectors[idx2][2]
                    sig2 = tuple(signs2[p] for p in pos_in_s2)
                    if sig2 not in s2_by_sig:
                        s2_by_sig[sig2] = []
                    s2_by_sig[sig2].append(idx2)
                    
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    sig1 = tuple(signs1[p] for p in pos_in_s1)
                    
                    if sig1 in s2_by_sig:
                        for idx2 in s2_by_sig[sig1]:
                            # Conflict
                            compat[idx1] &= ~(1 << idx2)
                            compat[idx2] &= ~(1 << idx1)
                            
            elif len(inter) == 4:
                # Same support
                # Conflict if signs are identical
                # If signs are different, they are compatible?
                # Dot product:
                # If signs match on 4, dot=4, dist^2=0. Conflict.
                # If signs match on 3, dot=2, dist^2=4. Compatible.
                # If signs match on 2, dot=0, dist^2=8. Compatible.
                # So only identical vectors conflict.
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    for idx2 in vecs_s2:
                        if idx1 == idx2: continue
                        signs2 = all_vectors[idx2][2]
                        if signs1 == signs2:
                            # Identical vectors are not possible in our generation
                            # because each (support, signs) is unique.
                            pass
                        # Wait, we generated unique (support, signs).
                        # So no two vectors have the same support AND same signs.
                        # So no conflicts for len(inter)==4 except self.
                        pass

    # 3. Max Clique Search on Compatibility Graph
    best_size = 0
    best_clique = 0
    
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    start_time = time.time()
    time_limit = 115.0

    # Greedy Coloring for Upper Bound
    # Colors are assigned to candidates.
    # The max color is an upper bound on the clique size.
    def greedy_color(candidates):
        # Returns a list of (vertex, color) sorted by color
        # Or just the max color and the list of vertices ordered by color
        uncolored = candidates
        color = 0
        order = []
        while uncolored:
            color += 1
            available = uncolored
            while available:
                # Pick a vertex, remove it and its neighbors from 'available' for this color?
                # No, greedy coloring:
                # Pick a vertex v from uncolored.
                # Assign color to v.
                # Remove v from uncolored.
                # Remove v's neighbors from the set of vertices that can take this color?
                # Standard greedy:
                # For each color c:
                #   Find a maximal independent set in the remaining graph? No.
                #   Find a vertex v, assign color c, remove v and its neighbors from the pool for color c?
                #   Actually, we want to color the graph such that adjacent vertices have different colors.
                #   So for a given color c, we can assign it to a set of non-adjacent vertices.
                #   We want to minimize the number of colors.
                #   Heuristic: Pick the vertex with highest degree first?
                
                # Simpler:
                # Iterate through vertices in some order.
                # Assign the smallest possible color.
                
                # For the bound, we just need a valid coloring.
                # We can use a simple sequential coloring.
                
                pass # Placeholder

    # Implementing a full Tomita algorithm with bitsets is complex.
    # Given the time, I will stick to the current MIS approach but improve the bound slightly or the heuristic.
    
    # Actually, the current approach is:
    # solve(candidates, current_mask, current_count)
    # Bound: current_count + popcount(candidates) <= best_size
    
    # I will implement a slightly better bound:
    # The maximum independent set size is at most the sum of 1/(1+deg) ? No.
    # I will use the "greedy independent set" size as a lower bound for pruning? No, we need upper bound.
    
    # Let's just optimize the existing code's performance and heuristics.
    # The current code is likely the bottleneck.
    # I will rewrite the solve function to be more efficient.

    def greedy_mis(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            min_deg = float('inf')
            v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = popcount(compat[u] & remaining) # Wait, we are doing MIS on Conflict Graph.
                # My `compat` is the Non-Conflict graph.
                # So `conflict[u]` is the Conflict graph.
                # I built `compat`. I should build `conflict` or use `~compat`.
                # Let's use `conflict` for clarity in the MIS solver.
                pass

    # Let's stick to the Conflict Graph MIS.
    conflict = [0] * N
    for i in range(N):
        conflict[i] = ((1 << N) - 1) ^ (1 << i) ^ compat[i]
        
    # Re-verify conflict construction
    # Conflict if NOT compatible.
    # Compatible if (i != j) and (not conflict).
    
    # Let's rebuild conflict directly to be safe.
    conflict = [0] * N
    for i in range(N):
        conflict[i] = 0
        
    for i_s, s1 in enumerate(supports):
        vecs_s1 = sup_to_vecs[s1]
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            vecs_s2 = sup_to_vecs[s2]
            
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                s2_by_sig = {}
                for idx2 in vecs_s2:
                    signs2 = all_vectors[idx2][2]
                    sig2 = tuple(signs2[p] for p in pos_in_s2)
                    if sig2 not in s2_by_sig:
                        s2_by_sig[sig2] = []
                    s2_by_sig[sig2].append(idx2)
                    
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    sig1 = tuple(signs1[p] for p in pos_in_s1)
                    
                    if sig1 in s2_by_sig:
                        for idx2 in s2_by_sig[sig1]:
                            conflict[idx1] |= (1 << idx2)
                            conflict[idx2] |= (1 << idx1)
                            
    # Now solve MIS on `conflict`
    
    def greedy_mis(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            min_deg = float('inf')
            v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = popcount(conflict[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
                temp ^= lsb
                
            if v == -1:
                break
                
            selected |= (1 << v)
            remaining &= ~(conflict[v] | (1 << v))
            
        return selected, popcount(selected)

    best_size = 0
    best_mask = 0

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
            
        # Branch on max degree vertex
        max_deg = -1
        v = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = popcount(conflict[u] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = u
            temp ^= lsb
            
        if v == -1:
            return
            
        v_bit = 1 << v
        
        new_candidates_1 = candidates & ~(conflict[v] | v_bit)
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
        result_vectors.append(all_vectors[i][3])
        temp ^= lsb
        
    return np.array(result_vectors, dtype=np.int64)
