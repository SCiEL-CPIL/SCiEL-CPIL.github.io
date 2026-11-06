import sys
import time

def construct():
    N = 256
    start_time = time.time()
    TIME_LIMIT = 55  # Leave 5s buffer
    
    # Build the non-conflict graph
    non_adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) != 0:
                non_adj[i] |= (1 << j)
                non_adj[j] |= (1 << i)
    
    def popcount(x):
        if x == 0:
            return 0
        return bin(x).count('1')
    
    # Greedy coloring of G (non_adj) for upper bound
    def greedy_color_bound(mask):
        if mask == 0:
            return 0
        
        colors = 0
        remaining = mask
        
        while remaining:
            colors += 1
            # Find maximal independent set in G induced by remaining
            independent_set = 0
            rem_for_is = remaining
            while rem_for_is:
                v_bit = rem_for_is & -rem_for_is
                v_idx = v_bit.bit_length() - 1
                independent_set |= v_bit
                rem_for_is &= ~(non_adj[v_idx] | v_bit)
            
            remaining &= ~independent_set
        
        return colors
    
    best_size = 0
    best_set = 0
    
    # Initial greedy solution to get a good lower bound
    # Greedy: repeatedly pick vertex with highest degree in remaining candidates
    def greedy_clique():
        candidates = (1 << N) - 1
        selected = 0
        size = 0
        while candidates:
            # Pick vertex with highest degree in candidates
            max_deg = -1
            v_idx = -1
            c = candidates
            while c:
                v_bit = c & -c
                idx = v_bit.bit_length() - 1
                deg = popcount(non_adj[idx] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    v_idx = idx
                c &= c - 1
            
            if v_idx == -1:
                break
            
            v_bit = 1 << v_idx
            selected |= v_bit
            size += 1
            candidates &= non_adj[v_idx]
        
        return selected, size
    
    init_set, init_size = greedy_clique()
    best_size = init_size
    best_set = init_set
    
    def branch(candidates, selected, size):
        nonlocal best_size, best_set
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if candidates == 0:
            if size > best_size:
                best_size = size
                best_set = selected
            return
        
        if size + popcount(candidates) <= best_size:
            return
        
        ub = size + greedy_color_bound(candidates)
        if ub <= best_size:
            return
        
        # Choose vertex with highest degree in candidates
        max_deg = -1
        v_idx = -1
        c = candidates
        while c:
            v_bit = c & -c
            idx = v_bit.bit_length() - 1
            deg = popcount(non_adj[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = idx
            c &= c - 1
        
        if v_idx == -1:
            return
        
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        new_candidates_include = candidates & non_adj[v_idx]
        branch(new_candidates_include, selected | v_bit, size + 1)
        
        # Branch 2: Exclude v_idx
        new_candidates_exclude = candidates & ~v_bit
        branch(new_candidates_exclude, selected, size)
    
    all_nodes = (1 << N) - 1
    branch(all_nodes, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    return result
