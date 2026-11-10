import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    # vertices 0..255 correspond to values 1..256
    
    # Precompute neighbors
    neighbors = [0] * N
    for i in range(N):
        u = i + 1
        for j in range(i + 1, N):
            v = j + 1
            g = math.gcd(u, v)
            x = u // g
            y = v // g
            if g % (x + y) != 0:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
                
    # Order by degree
    degrees = [bin(neighbors[i]).count('1') for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    new_index = [0] * N
    for new_idx, old_idx in enumerate(order):
        new_index[old_idx] = new_idx
        
    new_neighbors = [0] * N
    for i in range(N):
        old_i = order[i]
        mask = neighbors[old_i]
        new_mask = 0
        # Iterate bits
        m = mask
        while m:
            lsb = m & -m
            old_j = lsb.bit_length() - 1
            new_j = new_index[old_j]
            new_mask |= (1 << new_j)
            m ^= lsb
        new_neighbors[i] = new_mask

    best_clique = []
    best_size = 0

    def search(P, X, k):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if k > best_size:
                best_size = k
                best_clique = X[:]
            return
            
        if k + bin(P).count('1') <= best_size:
            return
            
        # Get vertices in P
        P_list = []
        p = P
        while p:
            lsb = p & -p
            idx = lsb.bit_length() - 1
            P_list.append(idx)
            p ^= lsb
            
        # Sort by degree in P
        # Optimization: Only sort if len(P_list) > 1
        if len(P_list) > 1:
            deg = {v: bin(new_neighbors[v] & P).count('1') for v in P_list}
            P_list.sort(key=lambda v: deg[v], reverse=True)
            
        # Coloring
        color = {}
        color_masks = []
        
        for v in P_list:
            c = -1
            for i in range(len(color_masks)):
                if (new_neighbors[v] & color_masks[i]) == 0:
                    c = i
                    break
            if c == -1:
                c = len(color_masks)
                color_masks.append(0)
                
            color[v] = c
            color_masks[c] |= (1 << v)
            
        # Branch
        P_local = P
        # Iterate in reverse order of P_list
        for v in reversed(P_list):
            if k + color[v] <= best_size:
                P_local &= ~(1 << v)
                continue
                
            new_P = P_local & new_neighbors[v]
            
            X.append(v)
            search(new_P, X, k + 1)
            X.pop()
            
            P_local &= ~(1 << v)
            
            if P_local == 0:
                break

    initial_P = (1 << N) - 1
    search(initial_P, [], 0)
    
    result = [v + 1 for v in best_clique]
    if not result:
        return [1]
    return result
