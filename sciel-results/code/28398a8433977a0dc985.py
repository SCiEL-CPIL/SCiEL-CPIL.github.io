import random
import time

def construct():
    n = 20
    # Precompute triples containing each vertex for fast checking?
    # Or just iterate.
    
    # To speed up, we can precompute the list of triples for each vertex u
    # triples_with[u] = list of tuples (a, b, c) where u is in the tuple
    # But checking bit_count is fast.
    
    def check_validity_with_edge(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        # We need to check all triples T such that u in T or v in T
        # A triple T is bad if |N(T)| >= 3
        # N(T) = {w | w connected to all x in T}
        
        # Iterate over all pairs (x, y) from V \ {u, v}
        # Triple T1 = (u, x, y)
        # Triple T2 = (v, x, y)
        # Triple T3 = (u, v, x)
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                
                # Triple (u, x, y)
                # Common neighbors: w such that w~u, w~x, w~y
                # Note: v is a candidate common neighbor for (u, x, y) if v~x and v~y
                # But we are checking if the CURRENT graph (with u-v added) has a bad triple.
                # So we just compute common = adj[u] & adj[x] & adj[y]
                # If bit_count(common) >= 3, invalid.
                
                common_uxy = adj[u] & adj[x] & adj[y]
                if common_uxy.bit_count() >= 3:
                    return False
                
                common_vxy = adj[v] & adj[x] & adj[y]
                if common_vxy.bit_count() >= 3:
                    return False
                
                common_uv_x = adj[u] & adj[v] & adj[x]
                if common_uv_x.bit_count() >= 3:
                    return False
                    
        return True

    def construct_greedy():
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not check_validity_with_edge(adj, u, v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        return adj

    best_adj = None
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Run many random restarts
    while time.time() < deadline:
        adj = construct_greedy()
        count = sum(adj[i].bit_count() for i in range(n)) // 2
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
