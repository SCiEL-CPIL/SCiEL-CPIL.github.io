import random
import time

def creates_c6(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    This is true if there is a simple path of length 5 between u and v in the current graph.
    Path: u - a - b - c - d - v
    """
    # Optimization: if u and v are already connected, this function shouldn't be called for that edge
    # But we call it for non-edges.
    
    # We need to find a, b, c, d such that:
    # a in N(u)
    # b in N(a) \ {u}
    # c in N(b) \ {u, a}
    # d in N(c) \ {u, a, b}
    # (d, v) in E and d != v
    
    neighbors_u = adj[u]
    neighbors_v = adj[v]
    
    # If u and v have no common neighbors, can they be part of a C6?
    # Path u-a-b-c-d-v.
    # a is neighbor of u. d is neighbor of v.
    # If N(u) and N(v) are disjoint, we still need the middle part b-c.
    
    for a in neighbors_u:
        if a == v:
            continue
        # b is neighbor of a, not u
        for b in adj[a]:
            if b == u or b == a:
                continue
            # c is neighbor of b, not u, a
            for c in adj[b]:
                if c == u or c == a or c == b:
                    continue
                # d is neighbor of c, not u, a, b
                # And d must be neighbor of v
                # So we just check neighbors of c
                for d in adj[c]:
                    if d == u or d == a or d == b or d == c:
                        continue
                    # Check if d is connected to v
                    if d in neighbors_v:
                        return True
    return False

def construct():
    n = 40
    best_edges = []
    best_m = 0
    start_time = time.time()
    timeout = 115  # Leave 5s for safety
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for trial in range(500):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with a random set of edges to avoid local minima of empty start
        # Try a few different initial densities
        # For this trial, pick a random initial size
        init_size = random.randint(20, 40)
        
        adj = [set() for _ in range(n)]
        
        # Add random initial edges
        random.shuffle(all_edges)
        for i in range(init_size):
            u, v = all_edges[i]
            # Check if this creates a C6 with existing edges?
            # Since we are starting small, it's unlikely to create a C6 immediately,
            # but to be safe, we can just add them. 
            # Actually, if we add edges that form a C6, the final graph is invalid.
            # So we must check.
            if not creates_c6(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
            else:
                # If it creates a C6, skip it (or try another)
                # For simplicity, skip.
                pass

        # Greedy addition of remaining edges
        # We need to consider all non-edges
        # We can iterate through all_edges again, but only those not in adj
        # To randomize the order of attempts, we shuffle a copy of the list of non-edges?
        # Or just shuffle all_edges and skip if already present.
        
        candidates = [e for e in all_edges if e[0] not in adj[e[1]]]
        random.shuffle(candidates)
        
        for u, v in candidates:
            # Double check if it's still a non-edge (it is, since we only add)
            if creates_c6(u, v, adj):
                continue
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            
        # Count edges
        m = sum(len(s) for s in adj) // 2
        if m > best_m:
            best_m = m
            # Construct edge list
            best_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        best_edges.append([u, v])
                        
    return best_edges
