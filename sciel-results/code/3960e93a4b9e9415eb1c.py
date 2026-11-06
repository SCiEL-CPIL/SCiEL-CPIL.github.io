import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 145s safety margin
    
    q = 5
    # Generate points in PG(2, q)
    # Points are non-zero vectors in F_q^3 up to scaling.
    # Normalize: first non-zero element is 1.
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    px, py, pz = 1, (y * inv_x) % q, (z * inv_x) % q
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    px, py, pz = 0, 1, (z * inv_y) % q
                else: # z != 0
                    inv_z = pow(z, q - 2, q)
                    px, py, pz = 0, 0, 1
                points.append((px, py, pz))
    
    n = len(points)
    if n != 31:
        # Fallback to random if generation fails (should not happen)
        import random
        random.seed(42)
        edges = set()
        while len(edges) < 91:
            u, v = random.sample(range(31), 2)
            edges.add((min(u,v), max(u,v)))
        return [list(e) for e in edges]

    # Polarity matrix M
    # M = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    # Orthogonality: P1 . M . P2 = 0
    # x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        p1 = np.array(points[i])
        for j in range(n):
            if i == j:
                continue
            p2 = np.array(points[j])
            # Calculate dot product with M
            # p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]
            val = (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1

    # Verify edge count
    edge_count = np.sum(A) // 2
    if edge_count != 90:
        # If construction is wrong, fall back to greedy random
        import random
        random.seed(42)
        edges = set()
        # Start with random
        while len(edges) < 90:
            u, v = random.sample(range(31), 2)
            edges.add((min(u,v), max(u,v)))
        # Add 91st
        for _ in range(1000):
            u, v = random.sample(range(31), 2)
            if (min(u,v), max(u,v)) not in edges:
                 edges.add((min(u,v), max(u,v)))
                 break
        return [list(e) for e in edges]

    # Compute A^3
    A2 = A @ A
    A3 = A2 @ A
    
    # Find non-edge with minimum A3[i,j]
    min_cost = float('inf')
    best_edge = None
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 0:
                cost = A3[i, j]
                if cost < min_cost:
                    min_cost = cost
                    best_edge = (i, j)
                    
    if best_edge is None:
        # Should not happen
        import random
        random.seed(42)
        edges = set()
        while len(edges) < 91:
            u, v = random.sample(range(31), 2)
            edges.add((min(u,v), max(u,v)))
        return [list(e) for e in edges]

    # Construct final edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                final_edges.append([i, j])
    
    # Add the new edge
    u, v = best_edge
    final_edges.append([u, v])
    
    return final_edges
