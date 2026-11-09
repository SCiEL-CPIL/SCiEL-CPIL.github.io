import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    max_n = 512
    
    # 1. Precompute adjacency bitmasks
    # adj[i] is a bitmask of nodes compatible with i
    adj = [0] * (max_n + 1)
    for i in range(1, max_n + 1):
        mask = 0
        for j in range(1, max_n + 1):
            if i == j:
                continue
            if (i * j) // math.gcd(i, j) <= max_n:
                mask |= (1 << j)
        adj[i] = mask
    
    # 2. Order vertices by degree (descending)
    # We want to map original indices to new indices 0..N-1
    degrees = [bin(adj[i]).count('1') for i in range(1, max_n + 1)]
    vertices = list(range(1, max_n + 1))
    vertices.sort(key=lambda x: degrees[x-1], reverse=True)
    
    # Map old index to new index
    old_to_new = {old: new for new, old in enumerate(vertices)}
    N = max_n
    
    # Build new adjacency bitmasks
    new_adj = [0] * N
    for i in range(N):
        old_i = vertices[i]
        mask = 0
        for j in range(N):
            old_j = vertices[j]
            if old_i == old_j:
                continue
            if (old_i * old_j) // math.gcd(old_i, old_j) <= max_n:
                mask |= (1 << j)
        new_adj[i] = mask
        
    # 3. Maximum Clique Search
    best_size = 0
    best_clique = []
    
    def expand(R, P):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        # Prune
        if len(R) + P.bit_count() <= best_size:
            return
            
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # Greedy Coloring
        # Returns list of colors for each node in P
        # and max_color
        colors = [0] * N
        max_color = 0
        U = P
        while U:
            max_color += 1
            V = U
            while V:
                # Pick a node, e.g., lowest set bit
                v = (V & -V).bit_length() - 1
                colors[v] = max_color
                V &= ~(1 << v)
                U &= ~(1 << v)
                V &= ~new_adj[v]
                
        # Bound check with coloring
        if len(R) + max_color <= best_size:
            return
            
        # Branching
        # Iterate nodes in P in reverse order of color
        # To do this, we can just iterate through bits of P
        # But Tomita's algorithm iterates through color classes
        # For simplicity, we pick the node with the highest color
        # Or we can iterate all nodes in P
        
        # Let's pick the node with the highest color
        # Actually, standard Tomita iterates from highest color to lowest
        # and within a color class, processes nodes
        
        # Simplified: Pick v with highest color in P
        v_max_color = -1
        v_best = -1
        P_temp = P
        while P_temp:
            v = (P_temp & -P_temp).bit_length() - 1
            if colors[v] > v_max_color:
                v_max_color = colors[v]
                v_best = v
            P_temp &= ~(1 << v)
            
        # Include v_best
        R.append(v_best)
        expand(R, P & new_adj[v_best])
        R.pop()
        
        # Exclude v_best
        P &= ~(1 << v_best)
        expand(R, P)

    # Initialize
    initial_P = (1 << N) - 1
    expand([], initial_P)
    
    # Map back to original indices
    result = [vertices[i] for i in best_clique]
    
    # Fallback if something went wrong
    if not result:
        result = [1]
        
    return result
