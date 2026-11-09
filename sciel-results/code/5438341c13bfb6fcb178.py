    def is_q3_subgraph(vertices, adj_matrix):
        v_map = {v: i for i, v in enumerate(vertices)}
        local_adj = [set() for _ in range(8)]
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, v = vertices[i], vertices[j]
                if adj_matrix[u][v]:
                    local_adj[i].add(j)
                    local_adj[j].add(i)
                    edge_count += 1
        
        if edge_count < 12:
            return False
        
        # Check if Q3 is a subgraph of the induced subgraph.
        # This is equivalent to checking if there is a mapping f: {0..7} -> {0..7}
        # such that for all (u,v) in Q3_edges, (f(u), f(v)) is in local_adj.
        # We don't require non-edges to map to non-edges (subgraph, not induced).
        
        # Backtracking
        q3_adj = [set() for _ in range(8)]
        for u, v in q3_edges:
            q3_adj[u].add(v)
            q3_adj[v].add(u)
            
        mapping = [-1] * 8
        used = [False] * 8
        
        def backtrack(idx):
            if idx == 8:
                return True
            u = idx
            for v in range(8):
                if used[v]:
                    continue
                # Check if v can be image of u
                ok = True
                for u2 in q3_adj[u]:
                    if mapping[u2] != -1:
                        if v not in local_adj[mapping[u2]]:
                            ok = False
                            break
                if not ok:
                    continue
                mapping[u] = v
                used[v] = True
                if backtrack(idx + 1):
                    return True
                mapping[u] = -1
                used[v] = False
            return False
        
        # Optimization: Fix 0 to 0?
        # Q3 is vertex transitive.
        # So we can fix Q3 vertex 0 to Local vertex 0.
        # But local vertex 0 might not be in the "right" position?
        # No, vertex transitivity means any vertex can be mapped to any other.
        # So yes, fix 0 to 0.
        mapping[0] = 0
        used[0] = True
        return backtrack(1)
