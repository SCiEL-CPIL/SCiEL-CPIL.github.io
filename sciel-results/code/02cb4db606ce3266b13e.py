def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Precompute all subsets of size 8
    subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute Q3 edges for isomorphism check
    q3_edges = set()
    # Q3 is the cube graph. Vertices 0..7. Edges (u,v) if popcount(u^v)==1
    for u in range(8):
        for v in range(u+1, 8):
            if bin(u ^ v).count('1') == 1:
                q3_edges.add((u, v))
                
    def is_q3(vertices, adj):
        """
        Check if the induced subgraph on 'vertices' is isomorphic to Q3.
        vertices: tuple of 8 unique vertex labels from 0..15
        adj: 16x16 boolean matrix
        """
        # 1. Check edge count
        ec = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[vertices[i]][vertices[j]]:
                    ec += 1
        if ec != 12:
            return False
            
        # 2. Check 3-regularity
        for i in range(8):
            deg = 0
            for j in range(8):
                if i != j and adj[vertices[i]][vertices[j]]:
                    deg += 1
            if deg != 3:
                return False
                
        # 3. Backtracking isomorphism
        # Map Q3 vertex (0..7) to subset index (0..7)
        
        def bt(q3_v, mapping, used_subset_indices):
            if q3_v == 8:
                return True
            
            for s_idx in range(8):
                if s_idx in used_subset_indices:
                    continue
                
                # Check if s_idx is compatible with already mapped neighbors
                ok = True
                for prev_q3_v in range(q3_v):
                    if (prev_q3_v, q3_v) in q3_edges:
                        prev_s_idx = mapping[prev_q3_v]
                        if not adj[vertices[prev_s_idx]][vertices[s_idx]]:
                            ok = False
                            break
                    else:
                        prev_s_idx = mapping[prev_q3_v]
                        if adj[vertices[prev_s_idx]][vertices[s_idx]]:
                            ok = False
                            break
                            
                if ok:
                    mapping[q3_v] = s_idx
                    if bt(q3_v + 1, mapping, used_subset_indices | {s_idx}):
                        return True
                    del mapping[q3_v]
            return False

        # Try starting with Q3 vertex 0 mapped to each subset vertex
        for start_idx in range(8):
            mapping = {0: start_idx}
            if bt(1, mapping, {start_idx}):
                return True
        return False

    def find_q3(adj):
        """
        Returns a tuple of 8 vertices forming a Q3, or None.
        """
        for subset in subsets:
            if is_q3(subset, adj):
                return subset
        return None

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    # Initialize with complete graph
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = True
            adj[j][i] = True
            
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = count_edges(adj)
    best_adj = [row[:] for row in adj]
    
    # Local Search Loop
    while time.time() < deadline:
        # 1. Find a Q3
        q3_subset = find_q3(adj)
        
        if q3_subset:
            # Remove an edge from the Q3
            induced_edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[q3_subset[i]][q3_subset[j]]:
                        induced_edges.append((q3_subset[i], q3_subset[j]))
                        
            if induced_edges:
                u, v = random.choice(induced_edges)
                adj[u][v] = False
                adj[v][u] = False
        else:
            # No Q3 found. Try to add an edge.
            missing_edges = [(i, j) for i, j in all_edges if not adj[i][j]]
            if missing_edges:
                u, v = random.choice(missing_edges)
                # Try adding it
                adj[u][v] = True
                adj[v][u] = True
                
                # Check if this created a Q3
                if find_q3(adj) is not None:
                    # Revert
                    adj[u][v] = False
                    adj[v][u] = False
                else:
                    # Update best
                    ec = count_edges(adj)
                    if ec > best_edges:
                        best_edges = ec
                        best_adj = [row[:] for row in adj]
            else:
                pass

    # Return the best valid graph found
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                res.append([i, j])
                
    return res
