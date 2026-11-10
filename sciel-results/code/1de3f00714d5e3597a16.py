<<<<<<< SEARCH
def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 118
    
    n = 16
    num_original = 11
    num_extra = n - num_original
    
    g_adj = get_groetzsch_adj(num_original)
    
    best_edges = []
    best_count = 0
    
    dists = generate_blowup_distributions(num_extra, num_original)
    
    for dist in dists:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        sizes = [1 + d for d in dist]
        
        orig_map = [0] * n
        current_idx = 0
        for i in range(num_original):
            for _ in range(sizes[i]):
                orig_map[current_idx] = i
                current_idx += 1
        
        edges = []
        adj = [0] * n
        
        for i in range(num_original):
            for j in range(i+1, num_original):
                if g_adj[i] & (1 << j):
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            edges.append([u, v])
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
        
        if not is_triangle_free_bitmask(adj, n):
            continue
            
        if is_not_3_colorable_fast(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
                
    if best_edges:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        deadline = START_TIME + TIME_LIMIT
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        if (adj[u] & adj[v]) == 0:
                            candidates.append((u, v))
            
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > deadline:
                    break
                
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                current_edges.append([u, v])
                
                if is_not_3_colorable_fast(adj, n):
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
                    improved = True
                else:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    current_edges.pop()
                    
            if not improved:
                break
                
    if not best_edges:
        # Fallback
        best_edges = build_groetzsch_edges()
        
    return best_edges
=======
def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 118
    
    n = 16
    num_original = 11
    num_extra = n - num_original
    
    g_adj = get_groetzsch_adj(num_original)
    
    best_edges = []
    best_count = 0
    
    dists = generate_blowup_distributions(num_extra, num_original)
    random.shuffle(dists)
    
    for dist in dists:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        sizes = [1 + d for d in dist]
        
        orig_map = [0] * n
        current_idx = 0
        for i in range(num_original):
            for _ in range(sizes[i]):
                orig_map[current_idx] = i
                current_idx += 1
        
        edges = []
        adj = [0] * n
        
        for i in range(num_original):
            for j in range(i+1, num_original):
                if g_adj[i] & (1 << j):
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            edges.append([u, v])
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
        
        if not is_triangle_free_bitmask(adj, n):
            continue
            
        if is_not_3_colorable_fast(adj, n):
            seed_deadline = time.time() + 10
            if time.time() > TIME_LIMIT - 5:
                seed_deadline = TIME_LIMIT - 5
                
            found_edges, found_count = local_search(adj, edges, n, seed_deadline)
            
            if found_count > best_count:
                best_count = found_count
                best_edges = list(found_edges)
                
    if not best_edges:
        best_edges = build_groetzsch_edges()
        
    return best_edges
>>>>>>> REPLACE
