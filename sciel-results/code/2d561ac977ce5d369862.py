import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1
    
    # 1. Identify valid nodes
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)
            
    num_valid = len(valid_nodes)
    
    # 2. Build adjacency for the COMPLEMENT graph (non-conflict graph)
    # adj_comp[i] is a bitmask of valid nodes j such that i+j is NOT a square.
    adj_comp = [0] * (N + 1)
    
    for a in valid_nodes:
        mask = 0
        for b in valid_nodes:
            if a != b and (a + b) not in squares:
                mask |= (1 << b)
        adj_comp[a] = mask
        
    best_clique_size = 0
    best_clique_mask = 0
    
    def color_candidates(P):
        if P == 0:
            return 0, []
        
        colors = {}
        max_color = 0
        remaining = P
        
        while remaining:
            max_color += 1
            U = remaining
            S = 0
            while U:
                lsb = U & (-U)
                v = lsb.bit_length() - 1
                S |= lsb
                U &= ~lsb
                U &= ~adj_comp[v]
            remaining &= ~S
            r = S
            while r:
                lsb = r & (-r)
                v = lsb.bit_length() - 1
                colors[v] = max_color
                r ^= lsb
                
        all_nodes = []
        r = P
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            all_nodes.append(v)
            r ^= lsb
            
        all_nodes.sort(key=lambda x: colors[x])
        
        return max_color, all_nodes

    def max_clique(P, current_clique_mask, current_size):
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() - start_time > deadline - 2:
            return
            
        if P == 0:
            if current_size > best_clique_size:
                best_clique_size = current_size
                best_clique_mask = current_clique_mask
            return
            
        if current_size >= num_valid:
            return

        max_color, ordered_nodes = color_candidates(P)
        
        if current_size + max_color <= best_clique_size:
            return

        for i in range(len(ordered_nodes) - 1, -1, -1):
            v = ordered_nodes[i]
            v_bit = 1 << v
            
            if not (P & v_bit):
                continue
                
            # Branch 1: Include v
            new_P = P & adj_comp[v]
            max_clique(new_P, current_clique_mask | v_bit, current_size + 1)
            
            # Branch 2: Exclude v
            P &= ~v_bit
            
            # If we have pruned enough, we might break early?
            # Not easily without re-coloring.
            
    max_clique(valid_mask_all, 0, 0)
    
    result = []
    m = best_clique_mask
    while m:
        lsb = m & (-m)
        node = lsb.bit_length() - 1
        result.append(node)
        m ^= lsb
        
    if not result:
        result = [1]
        
    res_set = set(result)
    for a in res_set:
        if 2 * a in squares:
            return [1]
        for b in res_set:
            if a <= b and (a + b) in squares:
                return [1]
                
    return result
