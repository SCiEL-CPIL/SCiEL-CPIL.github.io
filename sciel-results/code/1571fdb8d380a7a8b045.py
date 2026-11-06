import random
import itertools

def construct():
    n = 20
    vertices = list(range(n))
    best_edges_count = 0
    best_edge_list = []
    
    # Precompute all K3,3 structures (pairs of disjoint triples)
    triples = list(itertools.combinations(vertices, 3))
    k33_candidates = []
    for i, t1 in enumerate(triples):
        remaining = [v for v in vertices if v not in t1]
        for t2 in itertools.combinations(remaining, 3):
            if t1 < t2:
                k33_candidates.append((t1, t2))
                
    # Map edge to list of K33 indices
    edge_to_k33s = {}
    for idx, (t1, t2) in enumerate(k33_candidates):
        for u in t1:
            for v in t2:
                edge = (min(u, v), max(u, v))
                if edge not in edge_to_k33s:
                    edge_to_k33s[edge] = []
                edge_to_k33s[edge].append(idx)

    def greedy_deletion():
        # Returns a valid graph (adj matrix) and set of removed edges
        active_k33s = [True] * len(k33_candidates)
        active_count = len(k33_candidates)
        
        adj = [[True] * n for _ in range(n)]
        for i in range(n):
            adj[i][i] = False
            
        removed_edges = set()
        
        # To speed up, maintain a list of active K33 indices?
        # Or just scan. Scanning 100k items is fast.
        
        while active_count > 0:
            # Find an active K33
            target_idx = -1
            # Randomize start index to break symmetry?
            start = random.randint(0, len(k33_candidates)-1)
            for i in range(len(k33_candidates)):
                idx = (start + i) % len(k33_candidates)
                if active_k33s[idx]:
                    target_idx = idx
                    break
            
            if target_idx == -1:
                break
                
            t1, t2 = k33_candidates[target_idx]
            
            candidate_edges = []
            for u in t1:
                for v in t2:
                    edge = (min(u, v), max(u, v))
                    if adj[u][v]:
                        candidate_edges.append(edge)
                        
            if not candidate_edges:
                active_k33s[target_idx] = False
                active_count -= 1
                continue
                
            # Choose edge with max hits
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
                    
            if best_edge:
                u, v = best_edge
                adj[u][v] = False
                adj[v][u] = False
                removed_edges.add(best_edge)
                
                for k33_idx in edge_to_k33s.get(best_edge, []):
                    if active_k33s[k33_idx]:
                        active_k33s[k33_idx] = False
                        active_count -= 1
                        
        return adj, removed_edges

    def can_add_edge(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        # u in A, v in B => A={u, a1, a2}, B={v, b1, b2}
        # Need a1,a2 in N(v) and b1,b2 in N(u)
        # such that edges (a1,b1), (a1,b2), (a2,b1), (a2,b2) exist
        
        # Get neighbors
        # Note: adj[u][v] is currently False
        N_u = [x for x in range(n) if adj[u][x] and x != v]
        N_v = [x for x in range(n) if adj[v][x] and x != u]
        
        # We need to pick 2 from N_u (for B) and 2 from N_v (for A)
        # Check all pairs
        # Optimization: if len(N_u) < 2 or len(N_v) < 2, return True (safe)
        if len(N_u) < 2 or len(N_v) < 2:
            return True
            
        # Iterate pairs in N_u
        # To speed up, we can check if there is ANY K2,2 between N_v and N_u
        # For each pair (a1, a2) in N_v:
        #   Count how many nodes in N_u are connected to BOTH a1 and a2.
        #   If count >= 2, then we can pick two such nodes from N_u, and a K3,3 is formed.
        
        for a1, a2 in itertools.combinations(N_v, 2):
            # Find common neighbors of a1 and a2 within N_u
            common = 0
            for b in N_u:
                if adj[a1][b] and adj[a2][b]:
                    common += 1
            if common >= 2:
                return False # K3,3 would be formed
                
        return True

    def local_search(adj, removed_edges):
        # Try to add back removed edges
        # Randomize order
        removed_list = list(removed_edges)
        random.shuffle(removed_list)
        
        added_any = True
        while added_any:
            added_any = False
            # Shuffle again in each pass?
            random.shuffle(removed_list)
            for edge in removed_list:
                u, v = edge
                if not adj[u][v]: # Still removed
                    if can_add_edge(adj, u, v):
                        adj[u][v] = True
                        adj[v][u] = True
                        removed_edges.discard(edge)
                        added_any = True
                        # Note: adding an edge might prevent adding others later in the list
                        # But we continue the loop.
                        
        return adj, removed_edges

    # Main loop: Multiple restarts
    import time
    start_time = time.time()
    
    for restart in range(50): # Limit restarts
        if time.time() - start_time > 100:
            break
            
        adj, removed_edges = greedy_deletion()
        adj, removed_edges = local_search(adj, removed_edges)
        
        # Count edges
        edge_count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_edge_list = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
            
    return best_edge_list
