import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    
    # Helper: Check if graph is triangle-free
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v in adj[u] and u < v:
                    # Check common neighbors
                    # Actually, simpler: for each u, check if any pair in adj[u] is connected
                    pass
        
        # Efficient check using bitsets or sets
        # adj is list of sets
        for u in range(n):
            # If any two neighbors of u are connected, triangle exists
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True

    # Helper: Check if graph is 3-colorable using backtracking with heuristics
    def is_3_colorable(adj):
        # Use DSATUR or simple backtracking with ordering
        # For n=16, simple backtracking with static ordering is fast enough if graph is sparse
        # But we need it to be fast for many calls in local search.
        
        color = [-1] * n
        # Order vertices by degree descending (greedy heuristic for coloring)
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        def dfs(idx):
            if idx == n:
                return True
            
            u = order[idx]
            # Get available colors
            used_colors = set()
            for v in adj[u]:
                if color[v] != -1:
                    used_colors.add(color[v])
            
            for c in range(3):
                if c not in used_colors:
                    color[u] = c
                    if dfs(idx + 1):
                        return True
                    color[u] = -1
            return False
        
        return dfs(0)

    def build_ag5():
        """
        Construct Andrásfai graph AG(5) on 15 vertices (0..14).
        Vertices: (i, j) where i in 0..4, j in 0..2.
        Mapping: vertex = i * 3 + j
        Edges:
        (i, 0) ~ (j, 1) if (j - i) % 5 in {1, 2}
        (i, 1) ~ (j, 2) if (j - i) % 5 in {1, 2}
        (i, 2) ~ (j, 0) if (j - i) % 5 in {1, 2}
        """
        edges = []
        k = 5
        # Layers 0->1
        for i in range(k):
            for j in range(k):
                diff = (j - i) % k
                if diff in [1, 2]:
                    u = i * 3 + 0
                    v = j * 3 + 1
                    edges.append((u, v))
        
        # Layers 1->2
        for i in range(k):
            for j in range(k):
                diff = (j - i) % k
                if diff in [1, 2]:
                    u = i * 3 + 1
                    v = j * 3 + 2
                    edges.append((u, v))
        
        # Layers 2->0
        for i in range(k):
            for j in range(k):
                diff = (j - i) % k
                if diff in [1, 2]:
                    u = i * 3 + 2
                    v = j * 3 + 0
                    edges.append((u, v))
        
        return edges

    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def can_add_edge(adj, u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        # Check triangle
        # If any common neighbor, triangle formed
        if adj[u] & adj[v]:
            return False
        return True

    def try_extend_ag5():
        edges = build_ag5()
        adj = build_adj(edges)
        
        # Add vertex 15
        # We want to add as many edges from 15 to 0..14 as possible
        # The neighbors of 15 must form an independent set in the current graph.
        
        # Find max independent set in AG(5)
        # Greedy: pick vertex with min degree? Or just try to add edges one by one.
        
        # Let's try to add edges (15, v) for v in 0..14
        # We will use a backtracking search to find the max number of independent neighbors for 15.
        # Since n=15 is small, we can do a simple search.
        
        best_edges = list(edges)
        best_adj = [set(s) for s in adj]
        
        # Try all subsets of neighbors for 15? 2^15 = 32768. Feasible.
        # But we can be smarter.
        # Let's just iterate through vertices and add if possible, then try to improve.
        
        # Initialize 15 with no edges
        current_edges = list(edges)
        current_adj = [set(s) for s in adj]
        
        # Greedy add edges to 15
        candidates = list(range(15))
        random.shuffle(candidates)
        for v in candidates:
            if can_add_edge(current_adj, 15, v):
                current_edges.append((15, v))
                current_adj[15].add(v)
                current_adj[v].add(15)
        
        # Now, try to improve by swapping neighbors of 15
        # Or, run local search on the whole graph.
        
        # Local Search:
        # We have a valid graph (triangle-free, 4-chromatic? AG(5) is 4-chromatic. Adding 15 with edges might keep it 4-chromatic).
        # Check if current is 4-chromatic.
        if is_3_colorable(current_adj):
            # If it became 3-colorable, we need to fix it or restart.
            # But adding edges usually makes it harder to color.
            # If it's 3-colorable, we failed to maintain 4-chromaticity.
            # This shouldn't happen if we start with 4-chromatic and add edges, 
            # unless the added edges allow a 3-coloring that wasn't there before? 
            # No, adding edges restricts colorings. So if G is 4-chromatic, G+e is also 4-chromatic.
            # Wait, adding edges to a 4-chromatic graph keeps it 4-chromatic.
            # So current_adj should be 4-chromatic.
            pass

        # Now maximize edges.
        # We can try to add more edges between 0..14? No, AG(5) is maximal triangle-free?
        # AG(5) is not necessarily maximal triangle-free, but it's dense.
        # We can try to add edges between 0..14 if they don't form triangles.
        # And we can try to add more edges to 15.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding any possible edge
            for u in range(n):
                for v in range(u + 1, n):
                    if can_add_edge(current_adj, u, v):
                        # Add edge
                        current_edges.append((u, v))
                        current_adj[u].add(v)
                        current_adj[v].add(u)
                        improved = True
                        # Check if still 4-chromatic? Adding edges preserves 4-chromaticity.
                        # So we don't need to check 3-colorability after every add.
                        # We only need to check if we REMOVE an edge.
            
            # If no more edges can be added, we are at a local maximum.
            # To escape, we can try swapping edges.
            # Or, restart with different greedy order for vertex 15.
            
            # Let's do a simple restart if stuck
            if not improved:
                break

        return current_edges

    best_edges = None
    best_count = -1
    
    # Run multiple trials with different random seeds for the greedy part
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        random.seed(trial)
        edges = try_extend_ag5()
        
        # Verify validity
        adj = build_adj(edges)
        if is_triangle_free(adj) and not is_3_colorable(adj):
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = [list(e) for e in edges]
    
    if best_edges is not None:
        return best_edges
    
    # Fallback to Grötzsch + isolated vertices
    def groetzch_graph():
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        edges += [[5 + u, 10] for u in range(5)]
        return edges

    return groetzch_graph()
