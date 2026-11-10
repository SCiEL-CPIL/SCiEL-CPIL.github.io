<<<<<<< SEARCH
def construct():
    start = time.time()
    deadline = start + 110.0

    best_edges = None
    best_count = -1

    def time_left():
        return deadline - time.time()

    rng = random.Random(12345)

    # Local search: start from dense graphs, remove edges to break Q3s, then add back
    for trial in range(50):
        if time_left() < 10.0:
            break
        
        # Start with a random dense graph
        density = 0.6 + 0.3 * rng.random()
        adj = [0] * N
        edges = []
        for u, v in ALL_PAIRS:
            if rng.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        
        # Remove edges to break all Q3s
        # Iteratively find Q3s and remove an edge
        max_iter = 200
        for _ in range(max_iter):
            if time_left() < 5.0:
                break
            # Find a Q3 by checking all 8-vertex subsets
            found = False
            for vert_set in itertools.combinations(range(N), 8):
                vs = set(vert_set)
                # Check if induced subgraph on vs contains a Q3
                # Build induced subgraph
                ind_edges = [(u,v) for u,v in edges if u in vs and v in vs]
                if len(ind_edges) < 12:
                    continue
                # Check if there's a 3-regular spanning subgraph isomorphic to Q3
                # Simple check: each vertex has degree >= 3 in induced subgraph
                degs = {}
                for v in vert_set:
                    degs[v] = sum(1 for u in vert_set if u != v and (adj[v] >> u) & 1)
                if any(d < 3 for d in degs.values()):
                    continue
                # Try to find a Q3: use backtracking on 8 vertices
                # This is expensive but 8 is small
                # For now, just remove a random edge from the induced subgraph
                if ind_edges:
                    rem_u, rem_v = rng.choice(ind_edges)
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    edges.remove((rem_u, rem_v))
                    found = True
                    break
            if not found:
                break
        
        # Now try to add edges back
        count = len(edges)
        for u, v in ALL_PAIRS:
            if time_left() < 5.0:
                break
            if (adj[u] >> v) & 1:
                continue
            if not _creates_cube(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
                count += 1
        
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edges]

    if best_edges is None:
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    return best_edges
=======
def construct():
    start = time.time()
    deadline = start + 115.0

    best_edges = None
    best_count = -1

    def time_left():
        return deadline - time.time()

    rng = random.Random(12345)

    # Start from a known good structure: complete bipartite K_{4,12} has 48 edges
    # but likely contains Q3. Let's try to build up from sparse and add edges greedily.
    
    # Strategy: Start with empty graph, add edges one by one in random order,
    # skipping any edge that would create a Q3.
    # Then do local search: try to swap edges or add more.
    
    for trial in range(30):
        if time_left() < 15.0:
            break
        
        # Randomize edge order
        edge_order = list(ALL_PAIRS)
        rng.shuffle(edge_order)
        
        adj = [0] * N
        edges = []
        
        # Greedy add
        for u, v in edge_order:
            if time_left() < 10.0:
                break
            if not _creates_cube(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        
        count = len(edges)
        
        # Local search: try to add more edges
        improved = True
        while improved and time_left() > 10.0:
            improved = False
            for u, v in edge_order:
                if time_left() < 10.0:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if not _creates_cube(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    count += 1
                    improved = True
        
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edges]

    if best_edges is None:
        # Fallback: complete bipartite K_{3,13} = 39 edges, likely Q3-free
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
        best_count = 39
    
    return best_edges
>>>>>>> REPLACE

