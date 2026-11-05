def construct():
    import time
    start_time = time.time()
    time_limit = 140.0

    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem_adj = adj[:]
        
        cliques = []
        active_mask = (1 << n) - 1
        
        def find_max_clique_in(rem_adj, n, active_mask):
            best_clique = []
            
            def color_sort(P_mask):
                verts = []
                m = P_mask
                while m:
                    lsb = m & -m
                    idx = (lsb.bit_length() - 1)
                    verts.append(idx)
                    m ^= lsb
                if not verts:
                    return []
                
                def deg_in_P(v):
                    return (rem_adj[v] & P_mask).bit_count()
                
                verts.sort(key=deg_in_P, reverse=True)
                
                v_color = {}
                colored_vertices = []
                for v in verts:
                    used_colors = set()
                    neigh = rem_adj[v] & P_mask
                    nm = neigh
                    while nm:
                        lsb = nm & -nm
                        u = (lsb.bit_length() - 1)
                        if u in v_color:
                            used_colors.add(v_color[u])
                        nm ^= lsb
                    c = 1
                    while c in used_colors:
                        c += 1
                    v_color[v] = c
                    colored_vertices.append((c, v))
                
                colored_vertices.sort(key=lambda x: x[0])
                return colored_vertices

            def bron_kerbosch(R, P, X):
                nonlocal best_clique
                if P == 0:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                colored = color_sort(P)
                for color, v in reversed(colored):
                    if len(R) + color < len(best_clique):
                        break
                    R.append(v)
                    P_new = P & rem_adj[v]
                    X_new = X & rem_adj[v]
                    bron_kerbosch(R, P_new, X_new)
                    R.pop()
                    P ^= (1 << v)
                    X |= (1 << v)
            
            bron_kerbosch([], active_mask, 0)
            return best_clique

        def is_clique(clique, rem_adj):
            for i, u in enumerate(clique):
                for v in clique[i+1:]:
                    if not (rem_adj[u] & (1 << v)):
                        return False
            return True

        def remove_clique_edges(clique, rem_adj):
            for i, u in enumerate(clique):
                for v in clique[i+1:]:
                    if rem_adj[u] & (1 << v):
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)

        def update_active_mask(rem_adj, n):
            new_active = 0
            for i in range(n):
                if rem_adj[i]:
                    new_active |= (1 << i)
            return new_active

        # Main greedy loop
        while active_mask:
            if time.time() - start_time > time_limit:
                break
            
            clique = find_max_clique_in(rem_adj, n, active_mask)
            
            if not clique or len(clique) < 2:
                has_edge = False
                m = active_mask
                while m:
                    lsb = m & -m
                    u = (lsb.bit_length() - 1)
                    if rem_adj[u] & active_mask:
                        has_edge = True
                        break
                    m ^= lsb
                if not has_edge:
                    break
                m = active_mask
                edge_found = None
                while m:
                    lsb = m & -m
                    u = (lsb.bit_length() - 1)
                    neighbors = rem_adj[u] & active_mask
                    if neighbors:
                        lsb_n = neighbors & -neighbors
                        v = (lsb_n.bit_length() - 1)
                        edge_found = [u, v]
                        break
                    m ^= lsb
                if edge_found:
                    clique = edge_found
                else:
                    break
            
            if not is_clique(clique, rem_adj):
                m = active_mask
                edge_found = None
                while m:
                    lsb = m & -m
                    u = (lsb.bit_length() - 1)
                    neighbors = rem_adj[u] & active_mask
                    if neighbors:
                        lsb_n = neighbors & -neighbors
                        v = (lsb_n.bit_length() - 1)
                        edge_found = [u, v]
                        break
                    m ^= lsb
                if edge_found:
                    clique = edge_found
                else:
                    break
            
            cliques.append(clique)
            remove_clique_edges(clique, rem_adj)
            active_mask = update_active_mask(rem_adj, n)
        
        # Fallback: cover remaining edges as pairs
        if active_mask:
            for u in range(n):
                neigh = rem_adj[u] & active_mask
                while neigh:
                    lsb = neigh & -neigh
                    v = (lsb.bit_length() - 1)
                    if u < v:
                        cliques.append([u, v])
                    neigh ^= lsb
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
        
        # Post-optimization: try to reduce clique count via local search
        # Idea: for each pair of cliques, check if their union is a clique
        # If so, merge them
        improved = True
        max_iters = 100
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > time_limit - 5:
                break
            
            new_cliques = list(cliques)
            changed = False
            for i in range(len(new_cliques)):
                if i >= len(new_cliques):
                    break
                for j in range(i + 1, len(new_cliques)):
                    if j >= len(new_cliques):
                        break
                    # Check if union of clique i and j is a clique
                    union = list(set(new_cliques[i] + new_cliques[j]))
                    if len(union) < 2:
                        continue
                    # Check if union is a clique in original adj
                    is_c = True
                    for a in range(len(union)):
                        for b in range(a+1, len(union)):
                            u, v = union[a], union[b]
                            if not (adj[u] & (1 << v)):
                                is_c = False
                                break
                        if not is_c:
                            break
                    if is_c:
                        # Merge: replace cliques i and j with union
                        # But we need to verify that the union covers exactly
                        # the edges that were in cliques i and j
                        # Actually, this is tricky. Let's just try it and verify later.
                        # For now, skip this approach as it's complex to verify
                        pass
            # Alternative: try to remove vertices from cliques if they don't cover unique edges
            # This is also complex. Let's try a simpler approach:
            # For each clique, check if removing a vertex still covers all its edges
            # with other cliques. If so, shrink the clique.
            for i in range(len(new_cliques)):
                c = new_cliques[i]
                if len(c) <= 2:
                    continue
                for v in c:
                    # Check if all edges incident to v in this clique are covered by other cliques
                    # or if v has no unique edges in this clique
                    # This is complex. Skip for now.
                    pass
        
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
