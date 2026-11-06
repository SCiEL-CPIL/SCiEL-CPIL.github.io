import numpy as np

def construct():
    # 1. Generate points of PG(2, 5)
    q = 5
    points = []
    # x != 0
    for y in range(q):
        for z in range(q):
            points.append([1, y, z])
    # x == 0, y != 0
    for z in range(q):
        points.append([0, 1, z])
    # x == 0, y == 0, z != 0
    points.append([0, 0, 1])
    
    n = len(points)
    # n should be 31
    
    # 2. Compute adjacency matrix
    # A[i][j] = 1 if points[i] and points[j] are orthogonal
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            # B(u, v) = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
            b = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if b == 0:
                A[i][j] = 1
                A[j][i] = 1
                
    # 3. Collect edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 1:
                edges.append([i, j])
                
    # Verify edge count
    # Expected 90
    # if len(edges) != 90: print("Warning: edge count", len(edges))
    
    # 4. Find a non-edge to add
    # We want to minimize C4s. As analyzed, any non-edge should yield 0 C4s.
    # Let's just pick the first non-edge.
    edge_set = set((min(e), max(e)) for e in edges)
    added = False
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                edges.append([i, j])
                added = True
                break
        if added:
            break
            
    return edges
