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
        # rem_adj is the graph of uncovered edges
        rem_adj = adj[:]
        
        cliques = []
        
        # Active vertices bitmask
        active_mask = (1 << n) - 1
        
        while True:
            # Check time
            if time.time() - start_time > time_limit:
                # Fallback: cover remaining edges as single edges
                # This ensures validity
                break
                
            # Find max clique in rem_adj
            # We need a max clique function that works on rem_adj
            
            # To avoid modifying rem_adj inside the clique finder, we pass it.
            
            # Optimization: If the max clique size is 1, we can just pick an edge?
            # No, if max clique is 1, there are no edges?
            # If there are edges, max clique is at least 2.
            
            # We need to find the max clique in the subgraph defined by rem_adj.
            # The vertices are those that have at least one edge in rem_adj.
            
            # Let's implement a local max clique finder that uses rem_adj.
            
            def find_max_clique_in(rem_adj, n, active_mask):
                # Only consider vertices in active_mask
                # If active_mask is 0, return []
                
                best_clique = []
                
                def color_sort(P_mask):
                    # P_mask is subset of active_mask
                    # Get list of vertices
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
                        
                # Initial P is active_mask
                bron_kerbosch([], active_mask, 0)
                return best_clique

            clique = find_max_clique_in(rem_adj, n, active_mask)
            
            if not clique:
                # No more edges? Or something went wrong?
                # If active_mask is not 0 but no clique found, it means isolated vertices?
                # But we only keep active vertices with degree > 0.
                # If degree > 0, there is an edge, so clique of size 2 exists.
                # So clique should not be empty if active_mask has edges.
                
                # Check if active_mask has any edges
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
                
                # If has_edge but no clique found, something is wrong.
                # Fallback: pick any edge
                # Find an edge
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
            
            # Add clique to result
            cliques.append(clique)
            
            # Remove edges in clique from rem_adj
            for u in clique:
                for v in clique:
                    if u != v:
                        rem_adj[u] &= ~(1 << v)
                        
            # Update active_mask: remove vertices with degree 0 in rem_adj
            # It's cheaper to recompute active_mask from scratch or filter
            # Since n is small, recompute
            new_active = 0
            for i in range(n):
                if rem_adj[i] & new_active: # This is not correct, need to check against full new_active eventually
                    pass
            
            # Better:
            new_active = 0
            for i in range(n):
                if rem_adj[i]: # has at least one neighbor
                    new_active |= (1 << i)
            active_mask = new_active
            
        # If timed out, fallback to cover remaining edges
        if active_mask:
            # Cover remaining edges as pairs
            # To be efficient, just iterate
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
            
            # Update active mask again? Not needed if we clear rem_adj
            # But we need to make sure we don't add duplicates
            # The loop above adds [u,v] for u<v.
            # We should clear rem_adj for these edges.
            
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
