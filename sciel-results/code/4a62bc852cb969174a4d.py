import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 512
    adj = [0] * N
    
    for i in range(N):
        num_i = i + 1
        bits = 0
        for j in range(i + 1, N):
            num_j = j + 1
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= N:
                bits |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = bits
        
    order = list(range(N))
    order.sort(key=lambda x: adj[x].bit_count(), reverse=True)
    
    new_idx = [0] * N
    for new_i, old_i in enumerate(order):
        new_idx[old_i] = new_i
        
    new_adj = [0] * N
    for new_i in range(N):
        old_i = order[new_i]
        old_bits = adj[old_i]
        new_bits = 0
        b = old_bits
        while b:
            lsb = b & (-b)
            old_j = lsb.bit_length() - 1
            new_j = new_idx[old_j]
            new_bits |= (1 << new_j)
            b ^= lsb
        new_adj[new_i] = new_bits

    best_clique = []
    best_size = 0
    
    def color_sort(P_mask):
        # Get list of vertices
        P_list = []
        b = P_mask
        while b:
            lsb = b & (-b)
            v = lsb.bit_length() - 1
            P_list.append(v)
            b ^= lsb
            
        if not P_list:
            return [], [], 0
            
        # Sort by degree in P
        # To speed up, we can just use the global order?
        # No, degree in P is better.
        
        # Calculate degrees
        # This is O(N^2)
        # We can approximate by using global degree?
        # Let's try global degree first to save time.
        # P_list.sort(key=lambda x: new_adj[x].bit_count(), reverse=True)
        
        # Or just use the order they appear?
        # Let's use a simple greedy coloring without sorting by degree first,
        # but sorting the result by color.
        
        # Greedy Coloring
        # Assign colors to vertices in P_list
        # We want to minimize colors.
        
        # To speed up, we can use the fact that we just need an upper bound.
        # A valid coloring gives an upper bound.
        
        # Let's use a simple loop.
        v_color = {}
        # To make it faster, we can process vertices in order of decreasing degree in P.
        # Let's calculate degree in P once.
        degs = {}
        for v in P_list:
            degs[v] = (new_adj[v] & P_mask).bit_count()
            
        P_list.sort(key=lambda x: degs[x], reverse=True)
        
        for v in P_list:
            # Find neighbors in P that are already colored
            # Since we process in order, we can just check all neighbors in P
            # and see if they are in v_color
            
            used = set()
            nb_mask = new_adj[v] & P_mask
            b = nb_mask
            while b:
                lsb = b & (-b)
                u = lsb.bit_length() - 1
                if u in v_color:
                    used.add(v_color[u])
                b ^= lsb
                
            c = 1
            while c in used:
                c += 1
            v_color[v] = c
            
        # Sort vertices by color
        # Stable sort
        colored_vertices = sorted(P_list, key=lambda x: v_color[x])
        
        P_sorted = colored_vertices
        color_bound = [0] * len(P_sorted)
        max_c = 0
        for i, v in enumerate(P_sorted):
            c = v_color[v]
            if c > max_c:
                max_c = c
            color_bound[i] = max_c
            
        return P_sorted, color_bound, max_c

    def expand(P_mask, C_list):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(C_list) > best_size:
                best_size = len(C_list)
                best_clique = C_list[:]
            return
            
        P_sorted, color_bound, max_color = color_sort(P_mask)
        
        if len(C_list) + color_bound[-1] <= best_size:
            return
            
        for i in range(len(P_sorted) - 1, -1, -1):
            v = P_sorted[i]
            
            if len(C_list) + color_bound[i] <= best_size:
                break
                
            C_list.append(v)
            P_new = P_mask & new_adj[v]
            expand(P_new, C_list)
            C_list.pop()
            
            P_mask &= ~(1 << v)
            
            if time.time() > deadline:
                return

    full_mask = (1 << N) - 1
    expand(full_mask, [])
    
    result = [v + 1 for v in best_clique]
    if not result:
        result = [1]
    return result
