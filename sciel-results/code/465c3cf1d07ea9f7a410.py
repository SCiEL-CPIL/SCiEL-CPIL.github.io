"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    # Start with complete graph
    edges_set = set()
    for u in range(n):
        for v in range(u+1, n):
            edges_set.add((u, v))
    
    # Q3 structure: vertices 0..7, edges if xor in {1,2,4}
    q3_adj = {i: [] for i in range(8)}
    for i in range(8):
        for j in range(8):
            if i != j and (i ^ j) in (1, 2, 4):
                q3_adj[i].append(j)
                
    # Precompute degree counts for fast access? 
    # We'll update a degree array when removing edges.
    deg = [15] * n  # Complete graph on 16 vertices
    
    def has_q3(edge_set):
        """
        Find a Q3 subgraph in the graph defined by edge_set.
        Returns a list of edges forming the Q3, or None.
        """
        # Order Q3 vertices for backtracking to improve pruning
        # 0 is connected to 1, 2, 4
        # 1 is connected to 0, 3, 5
        # 2 is connected to 0, 3, 6
        # 4 is connected to 0, 5, 6
        # 3 is connected to 1, 2, 7
        # 5 is connected to 1, 4, 7
        # 6 is connected to 2, 4, 7
        # 7 is connected to 3, 5, 6
        order = [0, 1, 2, 4, 3, 5, 6, 7]
        
        mapping = {}
        
        def backtrack(idx):
            if idx == 8:
                return True
            q_v = order[idx]
            # Candidates: graph vertices not yet used
            # We can filter candidates based on neighbors already mapped
            candidates = []
            for g_v in range(n):
                if g_v in mapping.values():
                    continue
                # Check compatibility with already mapped neighbors
                ok = True
                for q_nei in q3_adj[q_v]:
                    if q_nei in mapping:
                        g_nei = mapping[q_nei]
                        u, v = (g_v, g_nei)
                        if u > v: u, v = v, u
                        if (u, v) not in edge_set:
                            ok = False
                            break
                if ok:
                    candidates.append(g_v)
            
            # Heuristic: try candidates with higher degree first (more likely to be part of a dense subgraph)
            # This might help find a Q3 faster if one exists
            candidates.sort(key=lambda x: deg[x], reverse=True)
            
            for g_v in candidates:
                mapping[q_v] = g_v
                if backtrack(idx + 1):
                    return True
                del mapping[q_v]
            return False

        if backtrack(0):
            q3_edges = []
            for i in range(8):
                for j in q3_adj[i]:
                    if i < j:
                        u, v = mapping[i], mapping[j]
                        if u > v: u, v = v, u
                        q3_edges.append((u, v))
            return q3_edges
        return None

    # Greedy removal
    # We can run until no Q3 is found or timeout (but we hope it's fast)
    max_iterations = 1000
    for _ in range(max_iterations):
        q3_edges = has_q3(edges_set)
        if q3_edges is None:
            break
        
        # Find the edge in q3_edges to remove
        # Heuristic: remove edge with max degree sum (proxy for violating most clauses)
        best_edge = None
        best_score = -1
        for u, v in q3_edges:
            score = deg[u] + deg[v]
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge:
            u, v = best_edge
            edges_set.remove(best_edge)
            deg[u] -= 1
            deg[v] -= 1
            
    return [list(e) for e in edges_set]
# EVOLVE-BLOCK-END
