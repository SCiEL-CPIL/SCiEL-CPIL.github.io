# Inside the loop for adding a vertex:
S_mask = 0
active_mask = (1 << n) - 1

# Greedy Hitting Set
while True:
    # Find an independent set of size 9 in G[active_mask]
    # If none exists, we are done (alpha < 9)
    # We need a function to find an independent set of size k
    
    def find_independent_set(adj, n, k, active_mask):
        # Returns mask of independent set of size k or 0 if not found
        non_adj = [0] * n
        for i in range(n):
            non_adj[i] = active_mask & ~(1 << i) & ~adj[i]
            
        def dfs(mask, size, current_set):
            if size == k:
                return current_set
            if mask == 0:
                return 0
            if mask.bit_count() < k - size:
                return 0
            
            # Pick a vertex
            # Heuristic: pick vertex with min degree in mask?
            # To find a set, picking low degree might help?
            # Or just pick first available.
            v = (mask & -mask).bit_length() - 1
            v_bit = 1 << v
            
            # Try including v
            res = dfs(non_adj[v] & mask, size + 1, current_set | v_bit)
            if res:
                return res
            # Try excluding v
            res = dfs(mask & ~v_bit, size, current_set)
            if res:
                return res
            return 0
        
        return dfs(active_mask, 0, 0)

    I = find_independent_set(adj, n, 9, active_mask & ~S_mask)
    if I == 0:
        break # No independent set of size 9 in G\S
    
    # Pick a vertex from I to add to S
    # Heuristic: pick vertex in I that appears in many other independent sets?
    # Approximation: pick vertex with max degree in G?
    # Or just random.
    verts_in_I = []
    temp = I
    while temp:
        lsb = temp & -temp
        v = lsb.bit_length() - 1
        verts_in_I.append(v)
        temp ^= lsb
        
    v = random.choice(verts_in_I)
    S_mask |= (1 << v)
    
    # Check time
    if time.time() > deadline - 10:
        break

# After constructing S, verify alpha(G\S) < 9
# If valid, add vertex.
