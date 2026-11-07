"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def max_flow(s, t, graph, n_nodes):
        """
        Edmonds-Karp max flow.
        graph: dict of dict {u: {v: cap}}
        s, t: source and sink node indices in 0..n_nodes-1
        Returns max flow value and parent pointers.
        """
        # Deep copy graph for residual
        residual = {u: {v: c for v, c in vs.items()} for u, vs in graph.items()}
        
        parent = {}
        max_flow_val = 0
        
        while True:
            # BFS to find augmenting path
            visited = [False] * n_nodes
            queue = [s]
            visited[s] = True
            parent[s] = -1
            
            found = False
            while queue and not found:
                u = queue.pop(0)
                for v, cap in residual[u].items():
                    if not visited[v] and cap > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == t:
                            found = True
                            break
                        queue.append(v)
                        
            if not found:
                break
            
            # Find min capacity on path
            path_flow = float('inf')
            v = t
            while v != s:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u
            
            max_flow_val += path_flow
            
            # Update residual capacities
            v = t
            while v != s:
                u = parent[v]
                residual[u][v] -= path_flow
                if v not in residual[u]:
                    residual[v][u] = 0
                residual[v][u] = residual.get(v, {}).get(u, 0) + path_flow
                v = u
                
        return max_flow_val, parent

    def try_construct(order):
        """
        Greedily add vertices in 'order' to branches.
        Returns (branches, paths_dict)
        """
        B = []
        U = set() # Used internal vertices
        paths = {} # (u, v) -> [u, ..., v]
        
        # To speed up flow, we pre-allocate or just build dict each time.
        # N=24 is small.
        
        for v in order:
            if v in B or v in U:
                continue
            
            # If B is empty, we can just add v? 
            # No, we need at least 2 branches. 
            # If B is empty, adding v is trivial (no paths needed).
            # But we need to connect to future branches.
            # Actually, if B is empty, we just add v to B.
            if len(B) == 0:
                B.append(v)
                continue
            
            # Build flow graph
            # Nodes: 0..2n-1. 
            # x_in = 2*x, x_out = 2*x+1
            n_flow_nodes = 2 * n
            flow_graph = {i: {} for i in range(n_flow_nodes)}
            
            # Add vertex split edges
            for x in range(n):
                if x in B or x in U:
                    cap = 0
                else:
                    cap = 1
                # x_in -> x_out
                flow_graph[2*x][2*x+1] = cap
                
            # Add original edges
            for u, v_edge in edges:
                # u_out -> v_in
                if flow_graph[2*u+1].get(2*v_edge, 0) < 1:
                    flow_graph[2*u+1][2*v_edge] = 1
                # v_out -> u_in
                if flow_graph[2*v_edge+1].get(2*u, 0) < 1:
                    flow_graph[2*v_edge+1][2*u] = 1
            
            # Source: v_out
            s = 2 * v + 1
            
            # Sinks: u_in for u in B
            # Add super sink T
            T = n_flow_nodes
            flow_graph[T] = {}
            for u in B:
                flow_graph[2*u][T] = 1
                # Also need to allow flow from u_in to T
                # Wait, flow goes v_out -> ... -> u_in -> T
                # So edge u_in -> T is correct.
                
            n_flow_nodes += 1 # T
            
            flow_val, parent = max_flow(s, T, flow_graph, n_flow_nodes)
            
            if flow_val == len(B):
                # Extract paths
                # For each u in B, trace path from T back to s?
                # Or from s to T?
                # parent points from node to predecessor in BFS tree.
                # So parent[T] is the node before T.
                # We need to find paths for each u in B.
                # Since flow is 1 to each u_in -> T, we can trace back from T.
                # But T has multiple incoming edges? No, we added edges u_in -> T.
                # The residual graph will show which edges were used.
                # Actually, max_flow returns parent pointers for the LAST augmenting path found?
                # No, standard Edmonds-Karp accumulates flow. The `parent` dict is overwritten in each BFS.
                # We need to extract ALL paths.
                
                # Better approach: Run BFS in residual graph from T backwards?
                # Or, since we know flow is 1 for each u, we can just find a path from s to u_in in the residual graph?
                # No, the residual graph changes.
                
                # Let's re-run BFS in the FINAL residual graph?
                # No, the `parent` dict from the last iteration only gives one path.
                
                # We need to modify max_flow to return all paths or use a different method.
                # Since N is small, we can just run a simple DFS/BFS in the residual graph 
                # to find a path from s to each u_in, and remove it?
                # But we already computed max flow.
                
                # Alternative: Just use the fact that we can find disjoint paths.
                # We can find one path from s to u1_in, then remove it from residual, find path to u2_in, etc.
                # This is essentially what max flow does, but we need to store them.
                
                # Let's rewrite the flow extraction.
                # We'll just run a BFS from s to each u_in in the ORIGINAL graph 
                # but constrained by "available" vertices?
                # No, we need the specific disjoint paths.
                
                # Simplest fix: Modify max_flow to store all augmenting paths.
                # Or, since we only need to know IF it's possible, and we need the paths,
                # we can do this:
                # 1. Check if max flow == len(B).
                # 2. If yes, find the paths by tracing back from T in the residual graph?
                #    The residual graph has edges with positive capacity.
                #    The path s -> ... -> T used in the last augmentation is in `parent`.
                #    But we need ALL paths.
                
                # Let's just re-run a simple search for each u in B, 
                # using the "used" vertices from previously found paths in THIS step.
                
                new_U = set()
                valid = True
                # We need to find |B| disjoint paths from v to each u in B.
                # We can do this greedily:
                # For each u in B:
                #   Find shortest path from v to u avoiding U_new and B (except v, u).
                #   If found, add internal vertices to U_new.
                #   If not, valid = False.
                
                # But this greedy path finding might fail even if flow exists.
                # However, for small N, it's a good heuristic.
                
                # Let's try to extract from the flow residual properly.
                # We'll just use the `parent` from the last BFS? No.
                
                # Okay, let's just do a simple BFS for each u, updating a local "blocked" set.
                local_blocked = set(U) # Start with global U
                # Note: v is the source, so it's not blocked for the start.
                # But v is in B? No, v is the new vertex.
                
                for u in B:
                    # BFS from v to u
                    # Avoid local_blocked
                    # v is allowed (start)
                    # u is allowed (end)
                    # B \ {v, u} are blocked (they are in local_blocked? No, they are in B, not U.
                    # We must avoid B \ {v, u} as internal vertices.
                    # So blocked = local_blocked | (set(B) - {v, u})
                    
                    blocked = local_blocked | (set(B) - {v, u})
                    
                    # BFS
                    q = [v]
                    prev = {v: None}
                    found = False
                    while q and not found:
                        curr = q.pop(0)
                        if curr == u:
                            found = True
                            break
                        for nb in adj[curr]:
                            if nb in blocked:
                                continue
                            if nb not in prev:
                                prev[nb] = curr
                                q.append(nb)
                    
                    if not found:
                        valid = False
                        break
                    
                    # Reconstruct path
                    path = []
                    curr = u
                    while curr is not None:
                        path.append(curr)
                        curr = prev[curr]
                    path.reverse()
                    
                    # Add internal vertices to local_blocked
                    for node in path[1:-1]:
                        local_blocked.add(node)
                        
                    # Store path
                    paths[(v, u)] = path
                
                if valid:
                    B.append(v)
                    U = local_blocked
            else:
                # Cannot add v
                pass
        
        return B, paths

    # Try different orderings
    best_B = []
    best_paths = {}
    
    # Ordering 1: Degree descending
    order1 = sorted(range(n), key=lambda x: -len(adj[x]))
    B1, P1 = try_construct(order1)
    if len(B1) > len(best_B):
        best_B = B1
        best_paths = P1
        
    # Ordering 2: Degree ascending
    order2 = sorted(range(n), key=lambda x: len(adj[x]))
    B2, P2 = try_construct(order2)
    if len(B2) > len(best_B):
        best_B = B2
        best_paths = P2
        
    # Ordering 3: Random (fixed seed for reproducibility)
    import random
    random.seed(42)
    order3 = list(range(n))
    random.shuffle(order3)
    B3, P3 = try_construct(order3)
    if len(B3) > len(best_B):
        best_B = B3
        best_paths = P3

    if len(best_B) < 2:
        # Fallback to edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Prepare output
    branches = [int(v) for v in best_B]
    out_paths = []
    from itertools import combinations
    for u, v in combinations(branches, 2):
        # Find path
        if (u, v) in best_paths:
            p = best_paths[(u, v)]
        elif (v, u) in best_paths:
            p = best_paths[(v, u)][::-1]
        else:
            # Should not happen
            p = [u, v]
        out_paths.append([int(x) for x in p])
        
    return {'branches': branches, 'paths': out_paths}

# EVOLVE-BLOCK-END
