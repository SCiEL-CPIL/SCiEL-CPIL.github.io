import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s margin

def _has_c6(adj):
    """
    Check if the graph represented by the boolean adjacency matrix 'adj' 
    contains a C6 (cycle of length 6).
    """
    n = adj.shape[0]
    # Precompute neighbor lists
    nbrs = [list(np.where(adj[i])[0]) for i in range(n)]
    
    for u in range(n):
        for v in range(u + 1, n):
            # Find all internal pairs (a, b) for paths u-a-b-v
            # a is neighbor of u, b is neighbor of v, a is neighbor of b
            # a != v, b != u, a != b
            paths = []
            # Optimization: iterate over smaller neighbor set?
            # Usually degrees are similar.
            for a in nbrs[u]:
                if a == v:
                    continue
                # b must be in N(v) and N(a)
                # We can intersect nbrs[v] and nbrs[a]
                # But we also need b != u
                # Let's just loop, it's small
                for b in nbrs[v]:
                    if b == u or b == a:
                        continue
                    if adj[a, b]:
                        paths.append((a, b))
            
            # Check for two vertex-disjoint internal pairs
            # (a1, b1) and (a2, b2) are disjoint if {a1, b1} and {a2, b2} are disjoint
            L = len(paths)
            for i in range(L):
                a1, b1 = paths[i]
                for j in range(i + 1, L):
                    a2, b2 = paths[j]
                    if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                        return True
    return False

def _count_edges(adj):
    return int(np.sum(adj)) // 2

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    
    n = 40
    best_adj = None
    best_m = 0
    
    # Multiple restarts to escape local optima
    # We use a budget of restarts based on time
    restarts = 0
    while time.time() - _START_TIME < _DEADLINE - 2.0:
        restarts += 1
        
        # Initialize a random graph
        # Strategy: Random regular graph or random graph with p=0.4
        # Random regular graph of degree d is often a good start.
        # Let's try d=5 or d=6.
        # Or just random graph with p=0.4 (approx 320 edges? No, 40*39/2 * 0.4 ~ 312 edges is too dense for C6-free max 105).
        # Max edges is 105. Total possible is 780. Density ~ 0.13.
        # So p=0.15 is a better start.
        
        rng = random.Random(restarts)
        adj = np.zeros((n, n), dtype=bool)
        m = 0
        
        # Generate random graph with p=0.15
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.15:
                    adj[i, j] = True
                    adj[j, i] = True
                    m += 1
        
        # Initial check: if it has C6, remove edges until it doesn't?
        # Or just start local search which will handle it?
        # Local search adds edges. If start has C6, we can't add.
        # So we must start with a C6-free graph or fix it.
        
        # Quick fix: while has_c6, remove random edge
        if _has_c6(adj):
            while _has_c6(adj):
                # Remove a random edge
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                if not edges:
                    break
                u, v = rng.choice(edges)
                adj[u, v] = False
                adj[v, u] = False
                m -= 1
        
        # Local Search
        # Try to add edges
        # We keep a list of non-edges to try
        # To speed up, we can shuffle non-edges and try them one by one
        
        while time.time() - _START_TIME < _DEADLINE - 2.0:
            # Collect non-edges
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i, j]:
                        non_edges.append((i, j))
            
            if not non_edges:
                break # Complete graph, shouldn't happen for C6-free on 40 nodes?
                # Actually K5 is C6-free, so 8 K5s is 80 edges.
                # Max is 105. So there are always non-edges.
                
            rng.shuffle(non_edges)
            
            added = False
            for u, v in non_edges:
                if time.time() - _START_TIME > _DEADLINE - 2.0:
                    break
                
                # Try adding (u, v)
                adj[u, v] = True
                adj[v, u] = True
                
                if not _has_c6(adj):
                    m += 1
                    added = True
                else:
                    adj[u, v] = False
                    adj[v, u] = False
            
            if m > best_m:
                best_m = m
                best_adj = adj.copy()
            
            if not added:
                # Stuck. Kick: remove a random edge
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                if edges:
                    u, v = rng.choice(edges)
                    adj[u, v] = False
                    adj[v, u] = False
                    m -= 1
                else:
                    break # Empty graph, restart
                    
                # Optional: after kick, do a few more add attempts before full restart?
                # The loop above will re-collect non-edges and try.
                
    # If best_adj is None, return seed result
    if best_adj is None:
        return [[5 * k + a, 5 * k + b] for k in range(8) for a in range(5) for b in range(a + 1, 5)]
        
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return edges
