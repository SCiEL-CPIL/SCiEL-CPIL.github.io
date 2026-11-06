import random
import time

def construct():
    start_time = time.time()
    # Use a safety margin. The limit is 150s.
    deadline = start_time + 145.0

    # We want to maximize n.
    # The problem asks for R(C4, K1,39) > n, which means a C4-free graph on n vertices
    # with min degree >= n - 39.
    # Known bounds suggest n=46 or 47 might be possible.
    # Let's try from high to low.
    
    best_n = 39
    best_edges = []
    
    # n=39 with min_deg=0 is always valid (empty graph).
    # n=40 requires min_deg=1.
    # ...
    # n=47 requires min_deg=8.
    
    # Try n from 47 down to 40
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
            
        min_deg_req = n - 39
        
        # If min_deg_req is 0, the empty graph works.
        if min_deg_req == 0:
            best_n = n
            best_edges = []
            return {'n': n, 'edges': best_edges}
            
        success = False
        edges_list = None
        
        # We will use a bitmask-based greedy construction.
        # adj[i] is an integer where bit j is set if edge (i, j) exists.
        
        # Number of restarts depends on time remaining.
        # Each restart is very fast with bitmasks.
        max_restarts = 20000
        restart_count = 0
        
        while restart_count < max_restarts:
            if time.time() > deadline:
                break
            restart_count += 1
            
            adj = [0] * n
            deg = [0] * n
            
            # We need to ensure min degree >= min_deg_req.
            # Greedy strategy:
            # 1. Start with empty graph.
            # 2. While min(deg) < min_deg_req:
            #    a. Find vertex u with minimum degree.
            #    b. Find a neighbor v such that adding (u,v) doesn't create a C4.
            #       To be "safe" and balance degrees, pick v with low degree.
            #       Or pick random safe v.
            #    c. If no safe v, restart.
            #    d. Add edge (u,v).
            
            # Optimization: Instead of finding min degree every time (O(n)),
            # we can maintain a list of vertices that need edges.
            # But for n<=47, O(n) is negligible.
            
            stuck = False
            # We need to add edges until all vertices have deg >= min_deg_req.
            # Note: Adding an edge increases degree of two vertices.
            
            # A better heuristic for success rate:
            # Pre-shuffle the order in which we pick vertices?
            # Or just always pick the one with lowest degree.
            
            # To add randomness, we can shuffle the vertex indices initially
            # or add random tie-breaking.
            
            # Let's try a variation:
            # Randomly select a vertex with deg < min_deg_req.
            # This avoids getting stuck in a local minimum where one vertex is always picked.
            
            while True:
                # Check if all degrees are sufficient
                # min(deg) < min_deg_req
                # Find a vertex with deg < min_deg_req
                candidates_u = [i for i in range(n) if deg[i] < min_deg_req]
                
                if not candidates_u:
                    break # Success! All vertices have sufficient degree.
                
                # Pick a random candidate u to avoid bias
                u = random.choice(candidates_u)
                
                # Find safe neighbors v for u
                safe_vs = []
                mask_u = adj[u]
                
                # Iterate through all v != u
                # We can iterate through bits of ~mask_u (excluding u)
                # But n is small, so just loop v in range(n)
                
                for v in range(n):
                    if v == u:
                        continue
                    if (mask_u >> v) & 1:
                        continue # Already connected
                    
                    # Check if adding (u, v) creates a C4.
                    # A C4 is u-a-b-v.
                    # a is in N(u), b is in N(v), and a~b.
                    # So we need to check if there exists a in N(u) such that
                    # N(a) intersects N(v).
                    # N(v) is adj[v].
                    # N(a) is adj[a].
                    # We need: (adj[u] & adj[v] & adj[a]) != 0 for some a in N(u)?
                    # No.
                    # The cycle is u - a - b - v - u.
                    # Edges: (u,a), (a,b), (b,v), (v,u).
                    # We are adding (u,v).
                    # We need to check if there is a path u-a-b-v of length 3.
                    # a is a neighbor of u.
                    # b is a neighbor of v.
                    # a and b are adjacent.
                    
                    # So:
                    # For each a in N(u):
                    #   If (adj[a] & adj[v]) != 0:
                    #       Then there is a b in N(a) that is also in N(v).
                    #       This forms a C4.
                    
                    is_safe = True
                    # Iterate through neighbors of u
                    mask_a_iter = mask_u
                    while mask_a_iter:
                        # Get lowest set bit
                        lsb = mask_a_iter & -mask_a_iter
                        a_idx = lsb.bit_length() - 1
                        mask_a_iter ^= lsb
                        
                        # Check if N(a) intersects N(v)
                        if adj[a_idx] & adj[v]:
                            is_safe = False
                            break
                    
                    if is_safe:
                        safe_vs.append(v)
                
                if not safe_vs:
                    stuck = True
                    break
                
                # Pick v
                # Heuristic: Pick v with lowest degree to balance
                # Or random?
                # Let's try a mix: 50% lowest degree, 50% random
                if random.random() < 0.5:
                    v = min(safe_vs, key=lambda x: deg[x])
                else:
                    v = random.choice(safe_vs)
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
            
            if not stuck:
                # Verify C4-free property to be absolutely sure
                # In a C4-free graph, any two vertices have at most 1 common neighbor.
                has_c4 = False
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            common = adj[i] & adj[j]
                            # If common has more than 1 bit set, there's a C4
                            # common & (common - 1) != 0 means more than 1 bit
                            if common & (common - 1):
                                has_c4 = True
                                break
                    if has_c4:
                        break
                
                if not has_c4:
                    # Verify min degree
                    if min(deg) >= min_deg_req:
                        success = True
                        # Extract edges
                        edges_list = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if (adj[i] >> j) & 1:
                                    edges_list.append([i, j])
                        break # Found a valid graph for this n
        
        if success:
            best_n = n
            best_edges = edges_list
            # Since we are iterating n from high to low, the first success is the best.
            # We can return immediately.
            return {'n': n, 'edges': best_edges}
            
    # If we failed to find a graph for n > 39, return the best we have.
    # The loop starts at 47 and goes down. If it didn't return, it means
    # we failed for all n >= 40.
    # So we fall back to n=39.
    return {'n': 39, 'edges': []}
