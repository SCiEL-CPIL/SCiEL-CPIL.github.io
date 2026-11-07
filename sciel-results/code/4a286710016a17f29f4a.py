import time

P = { ... } # Same as before

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    
    # 1. Adjacency
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    # 2. Precompute degrees in masks
    # deg[v][mask]
    deg = [[0] * (1 << n) for _ in range(n)]
    for v in range(n):
        dv = deg[v]
        av = adj[v]
        for mask in range(1, 1 << n):
            lsb = mask & -mask
            prev = mask ^ lsb
            u = lsb.bit_length() - 1
            if av & (1 << u):
                dv[mask] = dv[prev] + 1
            else:
                dv[mask] = dv[prev]
                
    # 3. Greedy Initialization
    # Try the first minv vertices
    init_mask = (1 << minv) - 1
    init_degs = set()
    for v in range(minv):
        init_degs.add(deg[v][init_mask])
    best_count = len(init_degs)
    best_mask = init_mask
    
    # 4. Exhaustive Search
    # We can iterate masks.
    # To speed up, we can skip masks with size < minv.
    for mask in range(1 << n):
        if time.time() > deadline:
            break
            
        # Quick size check
        # bit_count is fast
        if mask.bit_count() < minv:
            continue
            
        # Calculate distinct degrees using bitmask
        deg_mask = 0
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            d = deg[v][mask]
            deg_mask |= (1 << d)
            m ^= lsb
            
        c = deg_mask.bit_count()
        if c > best_count:
            best_count = c
            best_mask = mask
            if best_count == n: # Max possible
                break
                
    # 5. Extract vertices
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_vertices.append(int(v))
        m ^= lsb
        
    return best_vertices
