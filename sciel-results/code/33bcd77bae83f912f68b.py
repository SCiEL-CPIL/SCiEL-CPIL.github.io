"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_8subsets = list(combinations(range(n), 8))
    
    # Precompute: for each edge (u,v), the list of 8-subsets containing both u and v
    edge_to_subsets = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_to_subsets[(i,j)] = [vs for vs in all_8subsets if i in vs and j in vs]
    
    def is_q3(vs, adj):
        """Check if induced subgraph on vs (tuple of 8 vertices) is Q3."""
        ec = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[vs[i], vs[j]]:
                    ec += 1
        if ec != 12:
            return False
        for i in range(8):
            d = 0
            for j in range(8):
                if i != j and adj[vs[i], vs[j]]:
                    d += 1
            if d != 3:
                return False
        color = [-1]*8
        color[0] = 0
        stack = [0]
        while stack:
            u = stack.pop()
            for v in range(8):
                if v != u and adj[vs[u], vs[v]]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False
        return all(c >= 0 for c in color)
    
    # Start from K3,13
    adj = np.zeros((n, n), dtype=bool)
    for a in range(3):
        for b in range(3, 16):
            adj[a, b] = True
            adj[b, a] = True
    
    current_edges = 39
    best_edges = 39
    best_adj = adj.copy()
    
    # Get all absent edges
    absent = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
    random.shuffle(absent)
    
    # Try adding each absent edge
    for u, v in absent:
        if time.time() > deadline - 15.0:
            break
        
        # Check if adding (u,v) creates a Q3
        creates_q3 = False
        for vs in edge_to_subsets[(u,v)]:
            # Temporarily add the edge
            adj[u, v] = True
            adj[v, u] = True
            if is_q3(vs, adj):
                creates_q3 = True
                adj[u, v] = False
                adj[v, u] = False
                break
            adj[u, v] = False
            adj[v, u] = False
        
        if not creates_q3:
            adj[u, v] = True
            adj[v, u] = True
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
    
    # Second pass: try again with different order
    absent2 = [(i, j) for i in range(n) for j in range(i+1, n) if not best_adj[i, j]]
    random.shuffle(absent2)
    
    adj2 = best_adj.copy()
    current_edges2 = best_edges
    
    for u, v in absent2:
        if time.time() > deadline - 5.0:
            break
        creates_q3 = False
        for vs in edge_to_subsets[(u,v)]:
            adj2[u, v] = True
            adj2[v, u] = True
            if is_q3(vs, adj2):
                creates_q3 = True
                adj2[u, v] = False
                adj2[v, u] = False
                break
            adj2[u, v] = False
            adj2[v, u] = False
        
        if not creates_q3:
            adj2[u, v] = True
            adj2[v, u] = True
            current_edges2 += 1
            if current_edges2 > best_edges:
                best_edges = current_edges2
                best_adj = adj2.copy()
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
