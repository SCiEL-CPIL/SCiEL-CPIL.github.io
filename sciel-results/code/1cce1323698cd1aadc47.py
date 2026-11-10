"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Finds a dense triangle-free graph on 16 vertices with chromatic number >= 4.
    Uses exact backtracking to maximize the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit
    
    n = 16
    # Precompute all potential edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(edges)
    
    # Map edge index to (u, v)
    edge_to_uv = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_uv[idx] = (u, v)
        
    # For each vertex, list of incident edges
    incident_edges = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        incident_edges[u].append(idx)
        incident_edges[v].append(idx)
        
    # For each edge, list of edges that form a triangle with it
    # If (u,v) is present, any (u,w) or (v,w) cannot be present if (w,u) or (w,v) is present?
    # Actually, if (u,v) is in the graph, then for any w, we cannot have both (u,w) and (v,w).
    # So adding (u,v) forbids adding (u,w) if (v,w) is already present, and vice versa.
    
    # State:
    # edge_state: 0 = undecided, 1 = in graph, -1 = not in graph
    edge_state = [0] * m
    
    # For fast 3-colorability check, we can maintain the graph structure.
    # But doing a full 3-color check at every node is expensive.
    # We will only check 3-colorability when we have a "complete" candidate or at regular intervals?
    # No, the task says "exact backtracking". We need to be sure.
    # However, 16 vertices is small. A backtracking 3-color solver is very fast.
    
    best_edges_found = []
    
    # Initial best: Grötzsch graph (11 edges)
    # Vertices 0-4: Cycle
    # 5-9: Mycielski middle
    # 10: Mycielski top
    # 11-15: Isolated
    init_edges = []
    for i in range(5):
        init_edges.append([i, (i+1)%5])
    for u, w in [(i, (i+1)%5) for i in range(5)]:
        init_edges.append([u, 5+w])
        init_edges.append([w, 5+u])
    for u in range(5):
        init_edges.append([5+u, 10])
        
    best_count = len(init_edges)
    best_edge_set = set(tuple(sorted(e)) for e in init_edges)
    
    # Heuristic: Order edges to prioritize high-degree vertices? 
    # Or just randomize? Randomization helps avoid getting stuck in local optima of the search tree.
    # But for "maximize edges", a greedy approach with backtracking is good.
    
    # We can use a simple recursive backtracking that tries to add edges.
    # To ensure we find a *non-3-colorable* graph, we must verify at the end of each path.
    # Actually, we can prune if the graph is 3-colorable AND we can't add more edges? No.
    # If the graph is 3-colorable, it's invalid for the target property (chromatic number >= 4).
    # So we are searching for a triangle-free graph that is NOT 3-colorable.
    
    # Strategy:
    # 1. Generate a triangle-free graph.
    # 2. Check if it is 3-colorable.
    # 3. If not, update best.
    # 4. Backtrack.
    
    # To speed up, we can use the fact that if a graph is 3-colorable, adding edges might make it non-3-colorable.
    # But we must maintain triangle-free.
    
    def is_3_colorable(edge_list):
        # Simple backtracking 3-coloring
        adj = [[] for _ in range(n)]
        for u, v in edge_list:
            adj[u].append(v)
            adj[v].append(u)
            
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(colors[u] != c for u in adj[v]):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    # Optimized 3-colorability check using bitmasks or similar? 
    # For n=16, simple recursion is fine, but let's make it faster.
    
    def check_3_colorable_fast(adj_matrix):
        # adj_matrix is a list of lists of neighbors
        colors = [-1] * n
        
        # Order vertices by degree (descending) for faster pruning
        order = sorted(range(n), key=lambda x: -len(adj_matrix[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        # Re-map colors to order? No, just use the order for recursion.
        
        def rec(idx):
            if idx == n:
                return True
            v = order[idx]
            # Check available colors
            used = set()
            for u in adj_matrix[v]:
                if pos[u] < idx: # Only check already colored neighbors
                    used.add(colors[u])
            for c in range(3):
                if c not in used:
                    colors[v] = c
                    if rec(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        return rec(0)

    # To maintain triangle-free property efficiently:
    # When adding edge (u, v), we must ensure no w exists such that (u, w) and (v, w) are present.
    # We can precompute for each pair (u, v) the list of w's.
    # common_neighbors[u][v] = set of w such that (u,w) and (v,w) are potential edges.
    # If (u, v) is added, then for all w in common_neighbors[u][v], we cannot add both (u, w) and (v, w).
    # This constraint is dynamic.
    
    # Let's stick to a simpler check:
    # When considering adding edge (u, v):
    # 1. Check if it creates a triangle: for all w, if (u, w) is in graph AND (v, w) is in graph, then NO.
    # 2. If valid, add it, recurse, remove it.
    
    # To make this fast, we can maintain a matrix `in_graph[u][v]`.
    
    in_graph = [[False]*n for _ in range(n)]
    current_edges = []
    
    # Initialize with empty graph? Or start with a good seed?
    # Starting with empty graph and adding edges is a standard "maximal independent set" like search on the line graph of K_n?
    # No, it's finding a maximal triangle-free subgraph.
    
    # Let's use a randomized greedy approach with backtracking (simulated annealing-like or just random restarts).
    # Given the time limit, random restarts with greedy construction + local search might be better than pure backtracking which can be slow.
    # However, "exact backtracking" is in the prompt. Let's try to interpret it as "searching for the exact maximum".
    # With 120s, we can't do full exhaustive search of 2^120.
    # We must use pruning.
    
    # Pruning:
    # 1. Triangle-free constraint.
    # 2. 3-colorability check.
    
    # Let's try a recursive backtracking that picks edges in a specific order.
    # Order: Randomize edge order to escape local optima?
    # Or use a fixed order but with random restarts?
    
    import random
    
    best_score = best_count
    best_final_edges = init_edges[:]
    
    def search(edge_idx, current_edge_count):
        nonlocal best_score, best_final_edges
        
        if time.time() > deadline:
            return
            
        # Pruning: If current_edge_count + (m - edge_idx) <= best_score, we can't beat best.
        if current_edge_count + (m - edge_idx) <= best_score:
            return
            
        if edge_idx == m:
            # Check if the graph is 3-colorable
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for e in current_edges:
                u, v = e
                adj[u].append(v)
                adj[v].append(u)
                
            if not check_3_colorable_fast(adj):
                # Not 3-colorable, so chromatic number >= 4
                if current_edge_count > best_score:
                    best_score = current_edge_count
                    best_final_edges = current_edges[:]
                    # We can try to continue searching for even more?
                    # Yes, keep searching.
            return
            
        u, v = edges[edge_idx]
        
        # Option 1: Exclude edge (u, v)
        search(edge_idx + 1, current_edge_count)
        
        # Option 2: Include edge (u, v) if it doesn't create a triangle
        # Check for triangles
        valid = True
        for w in range(n):
            if w != u and w != v:
                if in_graph[u][w] and in_graph[v][w]:
                    valid = False
                    break
                    
        if valid:
            # Add edge
            in_graph[u][v] = True
            in_graph[v][u] = True
            current_edges.append((u, v))
            
            search(edge_idx + 1, current_edge_count + 1)
            
            # Remove edge
            current_edges.pop()
            in_graph[u][v] = False
            in_graph[v][u] = False
            
    # To improve performance, we can shuffle the edge order.
    # Let's do a few random restarts with different edge orderings.
    
    for restart in range(100):
        random.shuffle(edges)
        # Reset state
        in_graph = [[False]*n for _ in range(n)]
        current_edges = []
        
        # Run search
        search(0, 0)
        
        if time.time() > deadline:
            break
            
    return best_final_edges
# EVOLVE-BLOCK-END
