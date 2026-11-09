<<<<<<< SEARCH
    # Strategy 3: Start with a bipartite-like structure
    if time.time() < deadline:
        M = [0] * n
        # Split into two groups of 10
        group1 = list(range(10))
        group2 = list(range(10, 20))
        
        # Add edges within groups (sparse)
        for i in range(len(group1)):
            for j in range(i + 1, len(group1)):
                if random.random() < 0.3:
                    u, v = group1[i], group1[j]
                    if can_add(u, v):
                        add_edge(u, v)
        
        for i in range(len(group2)):
            for j in range(i + 1, len(group2)):
                if random.random() < 0.3:
                    u, v = group2[i], group2[j]
                    if can_add(u, v):
                        add_edge(u, v)
        
        # Add cross edges (denser)
        for u in group1:
            for v in group2:
                if random.random() < 0.6:
                    if can_add(u, v):
                        add_edge(u, v)
        
        try_add_all()
        local_search_maximize()
        
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
=======
    # Strategy 3: Start with a bipartite-like structure
    if time.time() < deadline:
        M = [0] * n
        # Split into two groups of 10
        group1 = list(range(10))
        group2 = list(range(10, 20))
        
        # Add edges within groups (sparse)
        for i in range(len(group1)):
            for j in range(i + 1, len(group1)):
                if random.random() < 0.3:
                    u, v = group1[i], group1[j]
                    if can_add(u, v):
                        add_edge(u, v)
        
        for i in range(len(group2)):
            for j in range(i + 1, len(group2)):
                if random.random() < 0.3:
                    u, v = group2[i], group2[j]
                    if can_add(u, v):
                        add_edge(u, v)
        
        # Add cross edges (denser)
        for u in group1:
            for v in group2:
                if random.random() < 0.6:
                    if can_add(u, v):
                        add_edge(u, v)
        
        try_add_all()
        local_search_maximize()
        
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
    
    # Strategy 4: Start from a nearly complete graph and remove edges to eliminate K3,3
    if time.time() < deadline:
        M = [0] * n
        # Start with all edges
        for u in range(n):
            for v in range(u + 1, n):
                add_edge(u, v)
        
        # Now remove edges to eliminate K3,3 subgraphs
        # Find triples with 3+ common neighbors and break them
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        break
                    for w in range(v + 1, n):
                        if time.time() > deadline:
                            break
                        common = M[u] & M[v] & M[w]
                        if common.bit_count() >= 3:
                            # Need to remove edges to reduce common neighbors
                            # Find a vertex x that is a common neighbor
                            x = (common & -common).bit_length() - 1  # Lowest set bit
                            # Try removing edge (u, x) or (v, x) or (w, x)
                            removed = False
                            for a in [u, v, w]:
                                if not removed and (M[a] >> x) & 1:
                                    remove_edge(a, x)
                                    # Check if this fixed the issue
                                    common_new = M[u] & M[v] & M[w]
                                    if common_new.bit_count() < 3:
                                        changed = True
                                        removed = True
                            if not removed:
                                # Try removing edge between two of u,v,w
                                for a, b in [(u, v), (v, w), (u, w)]:
                                    if (M[a] >> b) & 1:
                                        remove_edge(a, b)
                                        changed = True
                                        break
>>>>>>> REPLACE

