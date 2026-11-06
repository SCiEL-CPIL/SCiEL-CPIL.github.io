import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    N = limit
    # adj[i] is a bitmask where bit j is set if lcm(i, j+1) <= limit
    # Node i (1..N) corresponds to bit (i-1).
    adj = [0] * (N + 1)
    
    # Precompute compatibility masks
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i * j // gcd(i, j) <= limit:
                mask |= (1 << (j - 1))
        adj[i] = mask
        
    best_clique_size = 0
    best_clique = []
    
    def mask_to_list(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx + 1)
            m ^= lsb
        return verts

    def color_sort(P_mask):
        vertices = mask_to_list(P_mask)
        if not vertices:
            return []
        
        degs = []
        for v in vertices:
            d = (adj[v] & P_mask).bit_count()
            degs.append((d, v))
        
        degs.sort(key=lambda x: x[0], reverse=True)
        
        colors = {}
        max_color = 0
        
        for d, v in degs:
            neigh_mask = adj[v] & P_mask
            used_colors = set()
            m = neigh_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                u = idx + 1
                if u in colors:
                    used_colors.add(colors[u])
                m ^= lsb
            
            c = 1
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        ordered = []
        for v, c in colors.items():
            ordered.append((v, c))
        ordered.sort(key=lambda x: x[1], reverse=True)
        
        return ordered

    def expand(C, P_mask):
        nonlocal best_clique_size, best_clique
        
        if time.time() > deadline:
            return
        
        if not P_mask:
            if len(C) > best_clique_size:
                best_clique_size = len(C)
                best_clique = C[:]
            return
        
        ordered = color_sort(P_mask)
        remaining_mask = P_mask
        
        for v, c in ordered:
            if time.time() > deadline:
                return
            
            if len(C) + c <= best_clique_size:
                break
            
            C.append(v)
            
            new_P = remaining_mask & adj[v]
            
            expand(C, new_P)
            
            C.pop()
            
            remaining_mask &= ~(1 << (v - 1))
            
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << (i - 1))
        
    expand([], initial_P)
    
    if not best_clique:
        return [1]
        
    return sorted(best_clique)
