def construct():
    import time
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_triangle_free(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            if adj[u] & adj[v]:
                return False
        return True

    def is_3_colorable(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        color = [-1] * n
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                if all(color[nei] != c for nei in adj[v]):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def find_max_independent_set(adj, vertices):
        # Simple greedy/backtracking for MIS on small graphs
        # For 16 vertices, we can use a simple branch and bound or just greedy
        # Greedy: pick vertex with min degree in induced subgraph?
        # Let's use a simple backtracking for MIS size
        best_size = 0
        best_set = set()
        
        def rec(remaining, current):
            nonlocal best_size, best_set
            if not remaining:
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
                return
            # Pruning: if len(current) + len(remaining) <= best_size, return
            if len(current) + len(remaining) <= best_size:
                return
            
            # Pick a vertex
            v = next(iter(remaining))
            remaining.remove(v)
            
            # Include v
            # Neighbors of v in remaining cannot be included
            new_remaining = {u for u in remaining if u not in adj[v]}
            rec(new_remaining, current | {v})
            
            # Exclude v
            rec(remaining, current)
            
            remaining.add(v) # Restore
            
        rec(set(vertices), set())
        return best_set

    def greedy_complete(edge_set):
        edge_set = set(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    if not (adj[u] & adj[v]):
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        changed = True
        return edge_set

    # 1. Construct Grötzsch graph on vertices 0..10
    groetzsch_edges = set()
    for i in range(5):
        groetzsch_edges.add((i, (i+1) % 5))
    for i in range(5):
        u, w = i, (i+1) % 5
        groetzsch_edges.add((u, 5+w))
        groetzsch_edges.add((w, 5+u))
    for i in range(5):
        groetzsch_edges.add((5+i, 10))
        
    # Verify Grötzsch
    if not (is_triangle_free(groetzsch_edges) and not is_3_colorable(groetzsch_edges)):
        # Fallback if Grötzsch construction is flawed
        # Use a known non-3-colorable triangle-free graph or random search
        pass

    current_edges = groetzsch_edges
    current_vertices = list(range(11))
    
    # 2. Expand to 16 vertices
    for new_v in range(11, 16):
        # Build adjacency for current graph
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find Max Independent Set in current graph
        mis = find_max_independent_set(adj, set(current_vertices))
        
        # Connect new_v to all vertices in MIS
        for u in mis:
            current_edges.add((new_v, u))
            
        current_vertices.append(new_v)
        
        # Check if still valid (triangle-free and not 3-colorable)
        # Adding a vertex connected to an independent set preserves triangle-freedom.
        # Does it preserve non-3-colorability?
        # If G is not 3-colorable, and we add v connected to S (independent), 
        # any 3-coloring of G can be extended to v?
        # v needs a color different from all neighbors in S.
        # Since S is independent, they can all have the same color, say c1.
        # v can use c2 or c3.
        # So adding a vertex connected to an independent set might make it 3-colorable!
        # Example: Triangle-free non-3-colorable graph G. Add v connected to an independent set S.
        # If G is not 3-colorable, it has no 3-coloring.
        # The new graph G' has G plus v.
        # If G' is 3-colorable, then G is 3-colorable (restriction).
        # So G' is NOT 3-colorable.
        # Wait. If G is not 3-colorable, then no extension is 3-colorable.
        # So yes, it preserves non-3-colorability.

    # 3. Greedy completion
    final_edges = greedy_complete(current_edges)
    
    # 4. Final verification
    if is_triangle_free(final_edges) and not is_3_colorable(final_edges):
        return [list(e) for e in final_edges]
        
    # Fallback: Return Grötzsch + isolated vertices (if completion failed validity)
    # Or just return the constructed one if it's valid
    if is_triangle_free(current_edges) and not is_3_colorable(current_edges):
        return [list(e) for e in current_edges]
        
    # Last resort: Random search (same as original but simplified)
    import random
    random.seed(42)
    best_edges = set()
    for _ in range(1000):
        edge_set = set()
        adj = [set() for _ in range(n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if not (adj[u] & adj[v]):
                edge_set.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        if not is_3_colorable(edge_set):
            if len(edge_set) > len(best_edges):
                best_edges = edge_set
    return [list(e) for e in best_edges]
