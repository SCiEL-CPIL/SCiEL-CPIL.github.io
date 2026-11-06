import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 40
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) to graph adj creates a C6.
        A new C6 must use the new edge (u,v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a,b,c,d distinct and not u,v.
        """
        # a in N(u), b in N(a)\{u}, c in N(b)\{a,u}, d in N(c)\{b,a,u}, v in N(d)\{c,b,a,u}
        # and d != v
        Nu = adj[u]
        for a in range(n):
            if a == u or not Nu[a]:
                continue
            Na = adj[a]
            for b in range(n):
                if b in (u, a) or not Na[b]:
                    continue
                Nb = adj[b]
                for c in range(n):
                    if c in (u, a, b) or not Nb[c]:
                        continue
                    Nc = adj[c]
                    for d in range(n):
                        if d in (u, a, b, c) or not Nc[d]:
                            continue
                        if d == v or not adj[d, v]:
                            continue
                        return True
        return False
    
    def create_adj_mask(edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = adj[v, u] = True
        return adj
    
    def add_edge(adj, edges, u, v):
        adj[u, v] = adj[v, u] = True
        edges.append([u, v])
    
    def check_c6_with_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6, using a faster method."""
        # A new C6 must use the new edge (u,v).
        # Path: u - a - b - c - d - v
        # We need: a in N(u), d in N(v), and a path of length 3 from a to d
        # that avoids u, v and has distinct internal vertices.
        
        # For efficiency, precompute neighbors
        Nu = adj[u].nonzero()[0]
        Nv = adj[v].nonzero()[0]
        
        # For each a in N(u) and d in N(v), check if there's a path a-b-c-d
        # with b, c distinct and not in {u, v, a, d}
        
        # Better: for each a in N(u), for each d in N(v), a != d,
        # check if there exist b, c such that:
        #   b in N(a) \ {u, v, d}, c in N(b) \ {u, v, a, d}, c in N(d) \ {u, v, a, b}
        
        for a in Nu:
            if a == v:
                continue
            Na = adj[a].nonzero()[0]
            for d in Nv:
                if d == u or d == a:
                    continue
                Nd = adj[d].nonzero()[0]
                # Need b in N(a) \ {u, v, a, d} and c in N(b) ∩ N(d) \ {u, v, a, b, d}
                for b in Na:
                    if b in (u, v, a, d):
                        continue
                    Nb = adj[b].nonzero()[0]
                    # Check if any c in Nb ∩ Nd, c not in {u, v, a, b, d}
                    for c in Nb:
                        if c in (u, v, a, b, d):
                            continue
                        if adj[c, d]:
                            return True
        return False
    
    # Start with a structured seed: 8 copies of K5
    edges = []
    adj = np.zeros((n, n), dtype=bool)
    
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u, v = 5*k+a, 5*k+b
                adj[u, v] = adj[v, u] = True
                edges.append([u, v])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try to add all possible cross edges
    # Order: try edges between different K5 components first
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                continue
            if time.time() > deadline:
                return best_edges
            if not check_c6_with_edge(adj, i, j):
                adj[i, j] = adj[j, i] = True
                edges.append([i, j])
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
    
    # Second pass: try random orderings to potentially find more edges
    remaining = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                remaining.append((i, j))
    
    random.seed(42)
    for attempt in range(10):
        if time.time() > deadline:
            break
        random.shuffle(remaining)
        # Try adding edges in random order
        for i, j in remaining:
            if time.time() > deadline:
                break
            if not adj[i, j]:
                if not check_c6_with_edge(adj, i, j):
                    adj[i, j] = adj[j, i] = True
                    edges.append([i, j])
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = list(edges)
        # Reset adj to best state for next attempt
        # Actually, let's keep the best and try to improve from there
        # But we need to be careful: the adj matrix has changed
        # Let's just keep the best and break if we can't improve
        if len(edges) == best_count:
            break
    
    return best_edges
