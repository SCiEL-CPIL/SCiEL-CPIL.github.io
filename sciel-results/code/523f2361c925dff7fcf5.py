import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    
    # Check for bit_count
    if not hasattr(int, "bit_count"):
        def bit_count(x):
            return bin(x).count("1")
    else:
        def bit_count(x):
            return x.bit_count()
            
    # adj_H[v] is a bitmask of vertices compatible with v
    adj_H = [0] * (n + 1)
    
    for a in range(1, n + 1):
        mask = 0
        for b in range(1, n + 1):
            if a == b:
                continue
            if (a * b) % (a + b) != 0:
                mask |= (1 << (b - 1))
        adj_H[a] = mask
    
    best_clique = set()
    best_size = 0
    
    def greedy_initial_clique():
        candidates = (1 << n) - 2
        clique = set()
        while candidates:
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                c ^= lsb
                deg = bit_count(adj_H[v] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            if best_v == -1:
                break
            clique.add(best_v)
            candidates = candidates & adj_H[best_v]
            candidates &= ~(1 << (best_v - 1))
        return clique

    initial_clique = greedy_initial_clique()
    best_clique = initial_clique
    best_size = len(best_clique)
    
    def expand(candidates, current_clique, current_size):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if current_size + bit_count(candidates) <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = set(current_clique)
            return
            
        # Pick vertex with max degree
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            lsb = c & -c
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            c ^= lsb
            deg = bit_count(adj_H[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
        if best_v == -1:
            return
            
        v_bit = 1 << (best_v - 1)
        
        # Include
        new_candidates = candidates & adj_H[best_v]
        new_candidates &= ~v_bit
        current_clique.add(best_v)
        expand(new_candidates, current_clique, current_size + 1)
        current_clique.discard(best_v)
        
        # Exclude
        new_candidates = candidates & ~v_bit
        expand(new_candidates, current_clique, current_size)

    all_vertices_mask = (1 << n) - 2
    expand(all_vertices_mask, set(), 0)
    
    return sorted(best_clique)

construct()
