"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin
    
    n = 31
    q = 5
    target_edges = 91
    
    # Generate points of PG(2, q)
    points = []
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
            
    if len(points) != n:
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

    # Build adjacency matrix and list
    A = np.zeros((n, n), dtype=int)
    edges = []
    neighbors = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                A[i, j] = 1
                A[j, i] = 1
                neighbors[i].add(j)
                neighbors[j].add(i)
                edges.append((i, j))
                
    if len(edges) != 90:
        # Fallback if polarity graph construction fails or has wrong edge count
        # Just add random edges to reach 91
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            if (u, v) not in edges:
                edges.append((u, v))
                A[u, v] = 1
                A[v, u] = 1
                neighbors[u].add(v)
                neighbors[v].add(u)
        return [[u, v] for u, v in edges]

    def get_non_edges(A, neighbors):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 0:
                    non_edges.append((i, j))
        return non_edges

    def count_c4(A):
        # C4 = 1/2 * sum (A^2)^2 - sum A * A^2
        A2 = A @ A
        # Sum of squares of entries in A2
        s1 = np.sum(A2**2)
        # Sum of element-wise product of A and A2
        s2 = np.sum(A * A2)
        return (s1 - s2) // 2

    # Directly find the best edge to add to the polarity graph
    # The polarity graph is C4-free. Adding edge (u,v) creates P3(u,v) C4s.
    # We want to minimize P3(u,v).
    
    A2 = A @ A
    min_p3 = float('inf')
    best_add = None
    
    for x in range(n):
        for y in range(x+1, n):
            if A[x, y] == 0:
                # P3(x,y) is the number of common neighbors
                # Which is the (x,y) entry of A^2
                p3 = A2[x, y]
                if p3 < min_p3:
                    min_p3 = p3
                    best_add = (x, y)
                    
    if best_add:
        edges.append(best_add)
        
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
