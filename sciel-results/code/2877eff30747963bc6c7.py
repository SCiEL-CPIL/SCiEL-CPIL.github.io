import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256
    # 1. Build conflict masks
    conflict = [0] * N
    
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    for i in range(N):
        x = i + 1
        mask = 0
        for j in range(N):
            y = j + 1
            if x + y in squares:
                mask |= (1 << j)
        conflict[i] = mask
        
    # 2. Build compatibility masks
    all_mask = (1 << N) - 1
    compat = [0] * N
    for i in range(N):
        compat[i] = (all_mask ^ conflict[i]) & ~(1 << i)
        
    # 3. Pre-compute order for coloring (highest degree first)
    degrees = [compat[i].bit_count() for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    best_clique_size = 0
    best_clique = []
    
    # Greedy Initialization
    def greedy_clique():
        P = all_mask
        R = []
        while P:
            best_v = -1
            max_deg = -1
            for v in order:
                if P & (1 << v):
                    deg = (P & compat[v]).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
            if best_v == -1:
                break
            R.append(best_v)
            P &= compat[best_v]
        return R, len(R)
        
    init_R, init_size = greedy_clique()
    best_clique_size = init_size
    best_clique = init_R
    
    # Tomita's Algorithm
    def expand(R, P):
        nonlocal best_clique_size, best_clique
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique = R.copy()
            return
            
        current_P = P
        
        while current_P:
            if time.time() > deadline:
                return
                
            # Greedy Coloring of current_P using 'order'
            colors = [0] * N
            max_c = 0
            
            for v in order:
                if current_P & (1 << v):
                    neigh = current_P & compat[v]
                    used = 0
                    u = neigh
                    while u:
                        lsb = u & -u
                        u_idx = lsb.bit_length() - 1
                        c = colors[u_idx]
                        if c > 0:
                            used |= (1 << c)
                        u ^= lsb
                        
                    c = 1
                    while (used >> c) & 1:
                        c += 1
                        
                    colors[v] = c
                    if c > max_c:
                        max_c = c
                        
            if len(R) + max_c <= best_clique_size:
                return
                
            # Pick vertex from highest color class
            target_color = max_c
            v = -1
            for i in order:
                if current_P & (1 << i) and colors[i] == target_color:
                    v = i
                    break
            
            if v == -1:
                break
                
            # Branch
            new_R = R + [v]
            new_P = current_P & compat[v]
            
            expand(new_R, new_P)
            
            # Remove v from current_P
            current_P ^= (1 << v)
            
    initial_P = all_mask
    expand([], initial_P)
    
    result = [idx + 1 for idx in best_clique]
    
    # Final verification
    # squares_set = squares
    # for i in range(len(result)):
    #   for j in range(i, len(result)):
    #     if result[i] + result[j] in squares_set:
    #       return [1]
       
    return result
