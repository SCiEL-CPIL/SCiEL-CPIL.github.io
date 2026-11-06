"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin
    
    n = 31
    q = 5
    target_edges = 91
    
    # Generate points of PG(2, q)
    points = []
    # Points at infinity: [x:1:0] for x in 0..q-1, and [1:0:0]
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    # Finite points: [x:y:1] for x, y in 0..q-1
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
            
    if len(points) != n:
        # Fallback to simple random graph if something goes wrong with coordinates
        import random
        random.seed(42)
        edges = []
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            if (u, v) not in edges:
                edges.append((u, v))
        return [[u, v] for u, v in edges]

    # Build adjacency
    neighbors = [set() for _ in range(n)]
    edges = []
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            # Polarity condition: x1*x2 + y1*y2 - z1*z2 == 0 mod q
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
                
    current_edges = len(edges)
    
    # Find non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                non_edges.append((i, j))
                
    # We need to add (target_edges - current_edges) edges.
    # Usually this is 1.
    edges_to_add = target_edges - current_edges
    
    added_edges = []
    
    # Greedy addition of edges to minimize 4-cycles
    for _ in range(edges_to_add):
        if time.time() - start_time > DEADLINE:
            break
        best_edge = None
        min_cost = float('inf')
        
        for u, v in non_edges:
            # Check if this edge was already added in this loop (if edges_to_add > 1)
            if (u, v) in added_edges:
                continue
                
            # Cost = number of common neighbors u-x-v
            # This determines the number of new 4-cycles added: binom(cost, 2)
            cost = np.dot(adj[u], adj[v])
            
            if cost < min_cost:
                min_cost = cost
                best_edge = (u, v)
                
        if best_edge:
            u, v = best_edge
            edges.append((u, v))
            neighbors[u].add(v)
            neighbors[v].add(u)
            adj[u, v] = 1
            adj[v, u] = 1
            added_edges.append((u, v))
        else:
            break
            
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
