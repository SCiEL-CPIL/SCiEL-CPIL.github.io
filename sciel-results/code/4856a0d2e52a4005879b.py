import time
import random

def construct():
    deadline = time.time() + 110.0
    
    best_n = 0
    best_edges = []
    
    # Try to construct C4-free graphs with no independent set of size 11
    # Start from high n and work down
    
    for n in range(43, 29, -1):
        if time.time() > deadline - 25:
            break
        
        # Try multiple random orderings
        found_valid = False
        for seed in range(8):
            if time.time() > deadline - 25:
                break
            
            random.seed(seed * 7919 + n)
            adj = [0] * n
            edges = []
            
            # Greedy edge addition with C4 constraint
            order = list(range(n))
            random.shuffle(order)
            
            for i in range(n):
                u = order[i]
                for j in range(i):
                    v = order[j]
                    if adj[u] & (1 << v):
                        continue
                    # Check C4 constraint: |N(u) ∩ N(v)| <= 1
                    common = adj[u] & adj[v]
                    if common & (common - 1):
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
            
            # Check if independence number >= 11
            if not _has_independent_set(adj, n, 11, deadline):
                if n > best_n:
                    best_n = n
                    best_edges = edges[:]
                    found_valid = True
                    break
        
        if found_valid:
            break
    
    if best_n == 0:
        # Fallback: 30 triangles (C4-free, alpha = 10)
        best_n = 30
        best_edges = []
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            best_edges.append([u, v])
            best_edges.append([u, w])
            best_edges.append([v, w])
    
    return {'n': best_n, 'edges': best_edges}

def _has_independent_set(adj, n, k, deadline):
    """Check if graph has an independent set of size k."""
    deg = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda x: deg[x])
    
    found = [False]
    
    def bt(idx, size, forbidden):
        if found[0] or time.time() > deadline:
            return
        if size >= k:
            found[0] = True
            return
        if idx >= n or size + (n - idx) < k:
            return
        v = order[idx]
        if forbidden & (1 << v):
            bt(idx + 1, size, forbidden)
            return
        # Try including v
        new_forbidden = forbidden | (1 << v) | adj[v]
        bt(idx + 1, size + 1, new_forbidden)
        if not found[0]:
            # Try excluding v
            bt(idx + 1, size, forbidden)
    
    bt(0, 0, 0)
    return found[0]
