import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 40
    
    def now_dead():
        return time.time() >= deadline

    # 1. Initialization
    rng = np.random.default_rng(42)
    # Start with a moderately dense random graph
    p = 0.3
    adj = (rng.random((n, n)) < p).astype(int)
    np.fill_diagonal(adj, 0)
    adj = np.maximum(adj, adj.T)
    
    # Helper: Check if adding edge (u,v) creates a C6
    # A C6 using (u,v) requires a path of length 4 from u to v in G - {(u,v)}
    # Path: u-a-b-c-d-v
    def creates_c6(adj, u, v):
        # Get neighbors of u and v
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        # We need a path u-a-b-c-d-v
        # a in N(u), d in N(v)
        # b in N(a), c in N(d)
        # b ~ c
        # a != d, b != c, etc.
        
        # Optimization: 
        # For each a in N(u) \ {v}:
        #   For each d in N(v) \ {u, a}:
        #     Check if there is an edge between N(a) \ {u,v,d} and N(d) \ {u,v,a}
        
        for a in nu:
            if a == v: continue
            na = np.where(adj[a] == 1)[0]
            # Filter na: exclude u, v, d (d varies)
            # Actually, we need to check intersection of N(a) and N(d)
            # Let's iterate d
            for d in nv:
                if d == u or d == a: continue
                nd = np.where(adj[d] == 1)[0]
                
                # We need b in N(a) and c in N(d) such that b~c
                # b != u, v, a, d
                # c != u, v, a, d
                # b != c
                
                # Let's check if any node in N(a) is connected to any node in N(d)
                # Exclude u, v, a, d from the check?
                # If b=u, path is u-a-u... invalid.
                # So b must be in N(a) \ {u, v, a, d}
                # c must be in N(d) \ {u, v, a, d}
                
                # Create masks
                mask_a = np.ones(n, dtype=bool)
                mask_a[u] = False
                mask_a[v] = False
                mask_a[a] = False
                mask_a[d] = False
                candidates_a = na[mask_a]
                
                mask_d = np.ones(n, dtype=bool)
                mask_d[u] = False
                mask_d[v] = False
                mask_d[a] = False
                mask_d[d] = False
                candidates_d = nd[mask_d]
                
                if len(candidates_a) == 0 or len(candidates_d) == 0:
                    continue
                
                # Check if any edge exists between candidates_a and candidates_d
                # We can check if adj[candidates_a][:, candidates_d] has any 1
                if np.any(adj[np.ix_(candidates_a, candidates_d)] == 1):
                    return True
        return False

    # 2. Repair Phase: Remove edges that form C6s
    # To be fast, we iterate through edges and remove if they form a C6.
    # We repeat this until no more C6s are found or time runs out.
    # Since creates_c6 is O(n^3) roughly, and we have ~300 edges, it's fast.
    
    changed = True
    while changed and not now_dead():
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    if creates_c6(adj, u, v):
                        adj[u, v] = 0
                        adj[v, u] = 0
                        changed = True
    
    # 3. Local Search: Add edges
    best_adj = adj.copy()
    best_m = int(np.sum(adj) / 2)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while not now_dead():
        # Try to add a random edge
        u, v = all_pairs[rng.integers(0, len(all_pairs))]
        if adj[u, v] == 0:
            if not creates_c6(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                m = int(np.sum(adj) / 2)
                if m > best_m:
                    best_m = m
                    best_adj = adj.copy()

    # Return best_adj as list of edges
    best_edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                best_edges_list.append([u, v])
                
    return best_edges_list
