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

    # Initial best
    non_edges = get_non_edges(A, neighbors)
    best_c4 = float('inf')
    best_edges = None
    best_A = None
    
    # Find best edge to add to initial graph
    for u, v in non_edges:
        # Temporarily add
        A[u, v] = 1
        A[v, u] = 1
        c4 = count_c4(A)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = edges[:] + [(u, v)]
            best_A = A.copy()
        A[u, v] = 0
        A[v, u] = 0
        
    if best_A is None:
        # Should not happen
        return [[u, v] for u, v in edges]

    # Local search on the 90-edge base graph
    # We want to find a 90-edge C4-free graph that minimizes the min-P3 non-edge
    current_A = A.copy()
    current_edges = edges[:]
    
    # Precompute A2 for current graph
    current_A2 = current_A @ current_A
    
    # Helper to check if adding (u,v) creates C4 in graph with A
    # This is equivalent to checking if P3(u,v) == 0
    def creates_c4(A, A2, u, v):
        # P3(u,v) = sum_k A2[u,k] * A[k,v]
        # We can compute this quickly
        return np.dot(A2[u], A[:, v]) == 0

    # Search loop
    iterations = 0
    while time.time() - start_time < DEADLINE:
        iterations += 1
        # Pick a random edge to remove
        e_remove = random.choice(current_edges)
        u, v = e_remove
        
        # Find a random non-edge to add
        # We need a non-edge that does not create a C4
        # First, remove (u,v) from A
        current_A[u, v] = 0
        current_A[v, u] = 0
        # Recompute A2
        current_A2 = current_A @ current_A
        
        # Now find a non-edge (x,y) such that creates_c4(current_A, current_A2, x, y) is True
        # i.e. P3(x,y) == 0
        # We also need (x,y) to be a non-edge in current_A
        # And (x,y) should not be (u,v)
        
        # Let's try a few random non-edges
        found = False
        for _ in range(50): # Try 50 random non-edges
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if x > y: x, y = y, x
            if current_A[x, y] == 1: continue
            if (x, y) == (u, v): continue
            
            if creates_c4(current_A, current_A2, x, y):
                # Accept swap
                current_A[x, y] = 1
                current_A[y, x] = 1
                current_edges.remove(e_remove)
                current_edges.append((x, y))
                found = True
                break
                
        # Restore (u,v) if we didn't swap
        if not found:
            current_A[u, v] = 1
            current_A[v, u] = 1
            current_A2 = current_A @ current_A
            
        # Check if this new base graph yields a better 91-edge graph
        # Find min P3 non-edge in current_A
        # We need to add 1 edge to current_A to get 91 edges
        # We want to minimize the number of C4s in the resulting graph
        # This is equivalent to minimizing P3(x,y) for the added edge (x,y)
        
        min_p3 = float('inf')
        best_add = None
        # Iterate over non-edges of current_A
        for x in range(n):
            for y in range(x+1, n):
                if current_A[x, y] == 0:
                    # Compute P3(x,y)
                    p3 = np.dot(current_A2[x], current_A[:, y])
                    if p3 < min_p3:
                        min_p3 = p3
                        best_add = (x, y)
                        
        if best_add and min_p3 < best_c4:
            best_c4 = min_p3
            # Update best_A and best_edges
            best_A = current_A.copy()
            best_A[best_add[0], best_add[1]] = 1
            best_A[best_add[1], best_add[0]] = 1
            best_edges = current_edges[:] + [best_add]

        # Time check to ensure we return a valid result
        if time.time() - start_time > DEADLINE - 5.0:
            break

    # Return the best found
    if best_edges is None:
        # Fallback to initial graph + best single edge addition
        return [[u, v] for u, v in (edges + [best_add] if best_add else edges)]
        
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
