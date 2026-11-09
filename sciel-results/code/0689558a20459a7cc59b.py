import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    # Field GF(5)
    # Points of PG(2, 5)
    points = []
    # Affine part: z=1
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # Line at infinity: z=0
    for m in range(5):
        points.append((1, m, 0))
    points.append((0, 1, 0))
    
    n = len(points)
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            val = (points[i][0] * points[j][0] + 
                   points[i][1] * points[j][1] + 
                   points[i][2] * points[j][2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    # Find best edge to add
    min_c4s = float('inf')
    best_edge = None
    
    # Precompute common neighbors for all pairs
    # A2[i,j] = number of common neighbors of i and j
    A2 = A @ A
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                # Number of C4s created by adding (i,j) is the number of paths of length 3 between i and j
                # This is NOT just common neighbors.
                # It is the number of pairs (u,v) such that i-u-v-j is a path.
                # u is neighbor of i, v is neighbor of j, and u-v is an edge.
                # This is equivalent to the number of edges in the subgraph induced by N(i) and N(j)?
                # No.
                # Let's just add the edge, count C4s, and remove it.
                # Since n=31, this is fast.
                A[i, j] = 1
                A[j, i] = 1
                A2_new = A @ A
                c4s = 0
                for k in range(n):
                    for l in range(k + 1, n):
                        c = A2_new[k, l]
                        if c >= 2:
                            c4s += c * (c - 1) // 2
                c4s //= 2
                A[i, j] = 0
                A[j, i] = 0
                
                if c4s < min_c4s:
                    min_c4s = c4s
                    best_edge = (i, j)
                    
    # Add the best edge
    u, v = best_edge
    A[u, v] = 1
    A[v, u] = 1
    edges.append([u, v])
    
    # Local Search
    current_c4s = min_c4s
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_c4s = current_c4s
        
        # Try all edge swaps
        for e_idx in range(len(edges)):
            if time.time() > deadline:
                break
            eu, ev = edges[e_idx]
            
            # Remove edge
            A[eu, ev] = 0
            A[ev, eu] = 0
            
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if A[i, j] == 0 and not (i == eu and j == ev) and not (i == ev and j == eu):
                        A[i, j] = 1
                        A[j, i] = 1
                        
                        A2_new = A @ A
                        c4s = 0
                        for k in range(n):
                            for l in range(k + 1, n):
                                c = A2_new[k, l]
                                if c >= 2:
                                    c4s += c * (c - 1) // 2
                        c4s //= 2
                        
                        A[i, j] = 0
                        A[j, i] = 0
                        
                        if c4s < best_c4s:
                            best_c4s = c4s
                            best_swap = (e_idx, i, j)
                            
            A[eu, ev] = 1
            A[ev, eu] = 1
            
        if best_swap is not None:
            e_idx, new_i, new_j = best_swap
            old_u, old_v = edges[e_idx]
            edges[e_idx] = [new_i, new_j]
            A[old_u, old_v] = 0
            A[old_v, old_u] = 0
            A[new_i, new_j] = 1
            A[new_j, new_i] = 1
            current_c4s = best_c4s
            improved = True
    
    return edges
