"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_max_book(adj_set, n=40):
    """Compute the maximum book size (max triangles containing any edge)."""
    neighbors = [set() for _ in range(n)]
    for u, v in adj_set:
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    max_book = 0
    for (u, v) in adj_set:
        common_count = len(neighbors[u] & neighbors[v])
        if common_count > max_book:
            max_book = common_count
    return max_book

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 110.0  # Safety margin
    
    random.seed(42)
    
    # 1. Initial construction: near-regular graph via degree-constrained completion
    # Target degree ~ 401*2/40 = 20.05, so aim for degrees 20 or 21.
    # Start with a random regular-ish graph.
    
    # Generate a random graph with 401 edges, but bias towards regularity.
    # Strategy: Start with a cycle, then add random edges, but monitor degrees.
    # Better: Start with a specific structure.
    # K_20,20 has 400 edges, degrees 20. Add 1 edge -> one degree 21, one degree 21.
    # This is very regular.
    
    edges_set = set()
    # Bipartite K_20,20
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((a, b))
    
    # Add one intra-part edge to get to 401.
    # Choose (0,1) or similar.
    edges_set.add((0, 1))
    
    # Now we have 401 edges.
    # Degrees:
    # Vertices 0,1 in left part: deg 20 (to right) + 1 (to each other) = 21.
    # Other left vertices: deg 20.
    # Right vertices: deg 20.
    
    # This is a very good starting point.
    # Now use fast local search with numpy to improve.
    
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges_set:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Compute triangle counts (book sizes) for all pairs
    # tri[u, v] = number of common neighbors = (adj @ adj)[u, v]
    tri = (adj @ adj).astype(np.int32)
    
    # Current max book is max tri[u, v] for edges (u, v)
    # We can compute this quickly.
    current_max_book = 0
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                if tri[u, v] > current_max_book:
                    current_max_book = tri[u, v]
    
    best_adj = adj.copy()
    best_tri = tri.copy()
    best_max_book = current_max_book
    
    # Local search: swap edges to reduce max book.
    # We will maintain adj and tri.
    # A move: remove edge (u1, v1), add edge (u2, v2).
    # Update tri efficiently.
    
    # Precompute lists of edges and non-edges for sampling?
    # Or just sample randomly.
    
    # To speed up, we can identify "bad" edges (high book) and try to remove them.
    # Or just random swaps.
    
    # Let's try a hybrid:
    # 1. Random swap
    # 2. If no improvement for a while, target high-book edges.
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Perturb
            # Remove a random edge, add a random non-edge
            # Just do a few random swaps
            for _ in range(10):
                u1 = random.randint(0, n-1)
                v1 = random.randint(0, n-1)
                if u1 == v1 or not adj[u1, v1]:
                    continue
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2 or adj[u2, v2]:
                    continue
                
                # Swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                # Update tri
                # Remove (u1, v1)
                cn1 = np.where(adj[u1] & adj[v1])[0] # This is now AFTER removal? No, we need common neighbors BEFORE removal for the update logic?
                # Wait, tri[u, v] depends on neighbors.
                # If we remove (u1, v1), tri[u1, v1] becomes 0.
                # For w in N(u1) \ {v1}, tri[u1, w] changes?
                # tri[u1, w] = |N(u1) \cap N(w)|.
                # Removing v1 from N(u1) decreases tri[u1, w] by 1 if v1 in N(w).
                # v1 in N(w) is equivalent to w in N(v1).
                # So for w in N(v1) \ {u1}, tri[u1, w] -= 1 and tri[v1, w] -= 1?
                # Let's use the delta method from inspiration 2.
                
                # Actually, let's just recompute tri from scratch for simplicity if n=40 is small enough?
                # n=40, adj is 40x40. adj@adj is very fast.
                # 120s / (time for adj@adj) -> can do thousands of iterations.
                # Let's measure time for adj@adj. It's tiny.
                # So we can just do:
                # adj[u1, v1] = 0; adj[v1, u1] = 0
                # adj[u2, v2] = 1; adj[v2, u2] = 1
                # tri = (adj @ adj).astype(np.int32)
                # new_max = max(tri[u, v] for edges)
                
                # This is O(n^3) ~ 64000 ops. Very fast.
                # Let's do this. It's simpler and less error-prone.
                
                tri = (adj @ adj).astype(np.int32)
                new_max = 0
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v]:
                            if tri[u, v] > new_max:
                                new_max = tri[u, v]
                
                if new_max < best_max_book:
                    best_max_book = new_max
                    best_adj = adj.copy()
                    best_tri = tri.copy()
                    no_improve = 0
                else:
                    # Revert
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    no_improve += 1
            continue
        
        # Normal step: Random swap
        # Pick random edge
        while True:
            u1 = random.randint(0, n-1)
            v1 = random.randint(0, n-1)
            if u1 == v1: continue
            if adj[u1, v1]: break
        
        # Pick random non-edge
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2: continue
            if not adj[u2, v2]: break
        
        # Swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        tri = (adj @ adj).astype(np.int32)
        new_max = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    if tri[u, v] > new_max:
                        new_max = tri[u, v]
        
        if new_max < best_max_book:
            best_max_book = new_max
            best_adj = adj.copy()
            best_tri = tri.copy()
            no_improve = 0
        else:
            # Revert
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            no_improve += 1
    
    # Convert best_adj to list of edges
    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                edges_list.append([u, v])
    
    return edges_list

# EVOLVE-BLOCK-END
