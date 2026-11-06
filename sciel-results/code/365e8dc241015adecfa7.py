import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Check for bit_count availability
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    best_edge_count = 0
    best_adj = [0] * n
    
    def can_add(u, v, adj):
        # Check if adding (u,v) violates K2,3
        # For any w in N(v), if |N(u) \cap N(w)| >= 2, invalid
        # Because v will become a common neighbor of u and w (since w~v)
        # So new common count = |N(u) \cap N(w)| + 1
        # We need new count <= 2 => |N(u) \cap N(w)| <= 1
        
        nv = adj[v]
        mask = nv
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            if popcount(adj[u] & adj[w]) > 1:
                return False
            mask ^= lsb
            
        nu = adj[u]
        mask = nu
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            if popcount(adj[v] & adj[w]) > 1:
                return False
            mask ^= lsb
            
        return True

    def add_edge(u, v, adj):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Main loop
    while time.time() < deadline:
        adj = [0] * n
        current_edges = []
        
        # Greedy construction with random order
        # Using a shuffled copy of edges
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        for u, v in edges_copy:
            if can_add(u, v, adj):
                add_edge(u, v, adj)
                current_edges.append((u, v))
                
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
