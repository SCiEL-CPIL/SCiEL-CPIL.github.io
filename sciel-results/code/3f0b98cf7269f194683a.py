import itertools

def construct():
    n = 20
    vertices = list(range(n))
    
    # Precompute all triples
    triples = list(itertools.combinations(vertices, 3))
    
    # Generate all K3,3 candidates (pairs of disjoint triples)
    k33_candidates = []
    edge_to_k33s = {}
    
    # Map edge to list of k33 indices
    # Note: An edge (u,v) is in a K3,3 if u is in A and v in B (or vice versa)
    
    for i, t1 in enumerate(triples):
        # t1 is a tuple of 3 vertices
        # t2 must be disjoint from t1
        remaining = [v for v in vertices if v not in t1]
        for t2 in itertools.combinations(remaining, 3):
            # To avoid duplicates (A,B) and (B,A), ensure t1 < t2 lexicographically
            if t1 < t2:
                k33_candidates.append((t1, t2))
                
    # Build mapping from edge to list of K3,3 indices
    for idx, (t1, t2) in enumerate(k33_candidates):
        # Edges are all pairs (u, v) with u in t1, v in t2
        for u in t1:
            for v in t2:
                edge = (min(u, v), max(u, v))
                if edge not in edge_to_k33s:
                    edge_to_k33s[edge] = []
                edge_to_k33s[edge].append(idx)
    
    # Initial state: all edges present, all K3,3s active
    active_k33s = [True] * len(k33_candidates)
    active_count = len(k33_candidates)
    
    # Adjacency matrix for the graph
    adj = [[True] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = False
    
    # List of edges to remove
    removed_edges = set()
    
    # Helper to get active count for an edge
    # We need to know how many *active* K3,3s contain a given edge
    # We can just iterate through edge_to_k33s[edge] and check active_k33s[idx]
    
    # To speed up selection, we can just pick the first active K3,3
    # Then evaluate its edges
    
    while active_count > 0:
        # Find an active K3,3
        # Optimization: maintain a list of active indices
        # For simplicity, we scan. 387k is small.
        target_idx = -1
        for i in range(len(k33_candidates)):
            if active_k33s[i]:
                target_idx = i
                break
        
        if target_idx == -1:
            break
            
        t1, t2 = k33_candidates[target_idx]
        
        # Get the 9 edges of this K3,3
        candidate_edges = []
        for u in t1:
            for v in t2:
                edge = (min(u, v), max(u, v))
                # Only consider edges that are currently in the graph
                if adj[u][v]:
                    candidate_edges.append(edge)
                    
        if not candidate_edges:
            # Should not happen if active_k33s is correct, but safety check
            active_k33s[target_idx] = False
            active_count -= 1
            continue
            
        # Choose the edge that hits the most other active K3,3s
        best_edge = None
        max_hits = -1
        
        for edge in candidate_edges:
            hits = 0
            for k33_idx in edge_to_k33s.get(edge, []):
                if active_k33s[k33_idx]:
                    hits += 1
            if hits > max_hits:
                max_hits = hits
                best_edge = edge
                
        # Remove the best edge
        if best_edge:
            u, v = best_edge
            adj[u][v] = False
            adj[v][u] = False
            removed_edges.add(best_edge)
            
            # Update active K3,3s
            for k33_idx in edge_to_k33s.get(best_edge, []):
                if active_k33s[k33_idx]:
                    active_k33s[k33_idx] = False
                    active_count -= 1

    # Construct edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
