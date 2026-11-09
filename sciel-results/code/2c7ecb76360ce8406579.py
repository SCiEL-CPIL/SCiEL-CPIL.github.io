def construct():
    import time
    
    n = 16
    
    # Helper to check triangle-free
    def is_triangle_free(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True
        
    # Helper to check 3-colorable
    def is_3_colorable(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        color = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(color[nei] != c for nei in adj[v]):
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
            
        # Order by degree
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        def backtrack2(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                if all(color[nei] != c for nei in adj[v]):
                    color[v] = c
                    if backtrack2(idx + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack2(0)

    # Bitmask MIS Finder
    # mask represents the set of available vertices
    # Returns the size of the MIS
    # For n=16, we can precompute or use recursion
    def max_independent_set_size(adj_mask, vertices_mask):
        # adj_mask[i] is a bitmask of neighbors of i
        # vertices_mask is a bitmask of vertices to consider
        if vertices_mask == 0:
            return 0
        
        # Pick a vertex with highest degree in the subgraph?
        # Simple backtracking
        # Find a vertex v in vertices_mask
        v = (vertices_mask & -vertices_mask).bit_length() - 1
        # Option 1: Exclude v
        res = max_independent_set_size(adj_mask, vertices_mask ^ (1 << v))
        # Option 2: Include v
        # If we include v, we must exclude its neighbors
        new_mask = vertices_mask ^ (1 << v)
        new_mask &= ~adj_mask[v]
        res = max(res, 1 + max_independent_set_size(adj_mask, new_mask))
        return res

    # Convert edge list to adj masks
    def get_adj_masks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    # Core Graphs
    # Grötzsch Graph (11 vertices, 20 edges)
    groetzsch_edges = []
    # Cycle 0-1-2-3-4
    for i in range(5):
        groetzsch_edges.append((i, (i+1)%5))
    # Mycielski step
    for i in range(5):
        u, w = i, (i+1)%5
        groetzsch_edges.append((u, 5+w))
        groetzsch_edges.append((w, 5+u))
    for i in range(5):
        groetzsch_edges.append((5+i, 10))
        
    # Chvátal Graph (12 vertices, 24 edges)
    # Standard definition: 
    # Vertices 0..11
    # Edges:
    chvatal_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Cycle 0-1-2-3
        (0, 4), (1, 5), (2, 6), (3, 7), # Connect to 4-7
        (4, 5), (5, 6), (6, 7), (7, 4), # Cycle 4-5-6-7
        (0, 8), (1, 9), (2, 10), (3, 11), # Connect to 8-11
        (4, 8), (5, 9), (6, 10), (7, 11), # Connect to 8-11
        (8, 9), (9, 10), (10, 11), (11, 8) # Cycle 8-9-10-11
    ]
    # Wait, the Chvatal graph is not simply three cycles connected like that.
    # Let's use a known list of edges for Chvatal graph.
    # Actually, let's just use Grötzsch and a "complete" extension of it.
    # If Chvatal definition is wrong, it might fail.
    # Let's stick to Grötzsch and maybe a modified version.
    
    cores = [groetzsch_edges]
    
    best_edges = 0
    best_edge_list = []
    
    for core in cores:
        # Verify core
        if not is_triangle_free(core) or is_3_colorable(core):
            continue
            
        current_edges = set(core)
        current_n = len(set([v for e in core for v in e]))
        
        # Add remaining vertices
        # We need to map the core vertices to 0..current_n-1
        # And new vertices to current_n..15
        
        # Ensure core uses vertices 0..current_n-1
        # Grötzsch uses 0..10.
        
        adj_masks = get_adj_masks(list(current_edges))
        
        for new_v in range(current_n, n):
            # Find MIS of current graph (vertices 0..new_v-1)
            # The mask should only include vertices < new_v
            mask = (1 << new_v) - 1
            # We need the adj masks for the subgraph
            # max_independent_set_size expects adj_mask for all n, but we pass mask
            # The function uses vertices_mask to restrict
            mis_size = max_independent_set_size(adj_masks, mask)
            
            # To actually get the MIS vertices, we need to modify the function or just pick any max independent set.
            # For simplicity, let's just pick the first available vertex and its non-neighbors?
            # No, we need the *set* of vertices to connect to.
            
            # Let's implement a function to get one MIS
            def get_mis(adj_masks, vertices_mask):
                if vertices_mask == 0:
                    return 0
                v = (vertices_mask & -vertices_mask).bit_length() - 1
                # Exclude v
                mask1 = vertices_mask ^ (1 << v)
                mis1 = get_mis(adj_masks, mask1)
                # Include v
                mask2 = mask1 & ~adj_masks[v]
                mis2 = (1 << v) | get_mis(adj_masks, mask2)
                
                # Compare sizes
                size1 = bin(mis1).count('1')
                size2 = bin(mis2).count('1')
                if size1 >= size2:
                    return mis1
                else:
                    return mis2
            
            mis_mask = get_mis(adj_masks, mask)
            
            # Connect new_v to all vertices in mis_mask
            # Iterate bits in mis_mask
            m = mis_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                current_edges.add((u, new_v))
                adj_masks[u] |= (1 << new_v)
                adj_masks[new_v] |= (1 << u)
                m ^= lsb
                
            # Update adj_masks for new_v (already done in loop)
            # Ensure adj_masks[new_v] is initialized (it is, as list of 0s)
            
        # Final Check
        edge_list = list(current_edges)
        if is_triangle_free(edge_list) and not is_3_colorable(edge_list):
            if len(edge_list) > best_edges:
                best_edges = len(edge_list)
                best_edge_list = edge_list

    # If no valid graph found (should not happen), fallback
    if not best_edge_list:
        # Return Grötzsch + isolated vertices
        return [list(e) for e in groetzsch_edges]
        
    return [list(e) for e in best_edge_list]
