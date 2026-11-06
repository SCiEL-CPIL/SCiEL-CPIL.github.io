import sys

def construct():
    N = 256
    non_adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) != 0:
                non_adj[i] |= (1 << j)
                non_adj[j] |= (1 << i)
                
    best_size = 0
    best_set = 0
    
    def popcount(x):
        return x.bit_count()

    def greedy_color_bound(mask):
        if mask == 0:
            return 0
        
        colors = 0
        remaining = mask
        
        while remaining:
            colors += 1
            current_clique = 0
            rem_for_clique = remaining
            
            while rem_for_clique:
                # Pick vertex with highest degree in rem_for_clique
                max_deg = -1
                v_bit = 0
                v_idx = -1
                c = rem_for_clique
                while c:
                    lb = c & -c
                    idx = lb.bit_length() - 1
                    deg = popcount(non_adj[idx] & rem_for_clique)
                    if deg > max_deg:
                        max_deg = deg
                        v_bit = lb
                        v_idx = idx
                    c &= c - 1
                
                if v_idx == -1:
                    break

                current_clique |= v_bit
                # Keep only neighbors of v in G
                rem_for_clique &= non_adj[v_idx]
                rem_for_clique &= ~v_bit
                
            remaining &= ~current_clique
            
        return colors

    def branch(candidates, selected, size):
        nonlocal best_size, best_set
        
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
            
        # Choose a vertex to branch on
        # Highest degree in candidates
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
