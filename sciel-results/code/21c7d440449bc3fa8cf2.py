import random

def can_add(adj, u, v):
    # Check if adding edge (u, v) maintains K2,3-free property
    # 1. Common neighbors of u and v must be <= 2
    if (adj[u] & adj[v]).bit_count() > 2:
        return False
    
    # 2. For any neighbor w of v, the pair (u, w) must have <= 2 common neighbors in new graph.
    # In new graph, N'(u) = N(u) U {v}, N'(w) = N(w).
    # Intersection = (N(u) cap N(w)) U ({v} cap N(w)).
    # Since w is neighbor of v, v is in N(w). So v is in intersection.
    # Size = |N(u) cap N(w)| + 1.
    # We need |N(u) cap N(w)| + 1 <= 2 => |N(u) cap N(w)| <= 1.
    mask_v = adj[v]
    while mask_v:
        lsb = mask_v & -mask_v
        w = lsb.bit_length() - 1
        if (adj[u] & adj[w]).bit_count() > 1:
            return False
        mask_v ^= lsb
        
    # 3. Symmetrically for neighbors of u
    # For any neighbor w of u, pair (v, w) must have <= 2 common neighbors.
    # N'(v) = N(v) U {u}, N'(w) = N(w).
    # u is in N(w) since w is neighbor of u.
    # Size = |N(v) cap N(w)| + 1.
    # Need |N(v) cap N(w)| <= 1.
    mask_u = adj[u]
    while mask_u:
        lsb = mask_u & -mask_u
        w = lsb.bit_length() - 1
        if (adj[v] & adj[w]).bit_count() > 1:
            return False
        mask_u ^= lsb
        
    return True

def greedy_construct():
    adj = [0] * 20
    edges = []
    all_edges = [(u, v) for u in range(20) for v in range(u + 1, 20)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
    return edges

def construct():
    best_edges = []
    best_count = 0
    
    # Run multiple random greedy constructions to find a better solution
    # 2000 runs is fast and likely to find the optimum or near-optimum for N=20
    for _ in range(2000):
        edges = greedy_construct()
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
    return best_edges
