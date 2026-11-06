# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    q = 5
    n = 31
    
    # Enumerate points of PG(2,5)
    # 25 affine points (x, y, 1)
    # 5 points at infinity (1, t, 0)
    # 1 point at infinity (0, 1, 0)
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for t in range(q):
        points.append([1, t, 0])
    points.append([0, 1, 0])
    
    # Verify n
    assert len(points) == n
    
    # Build adjacency matrix
    # A[i][j] = 1 if dot(points[i], points[j]) % q == 0 and i != j
    A = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                A[i][j] = 1
                A[j][i] = 1
    
    # Count edges
    edge_count = np.sum(A) // 2
    
    # The polarity graph of PG(2,5) should have 90 edges and be C4-free.
    # We need 91 edges.
    # We add one edge (u, v) that is not currently present.
    # The number of new C4s created is the number of length-3 paths from u to v in the original graph.
    # This is given by (A^3)[u][v].
    
    # Compute A^3
    A2 = A @ A
    A3 = A2 @ A
    
    # Find the pair (u, v) with u < v, A[u][v] == 0, and minimal A3[u][v]
    min_c4s = float('inf')
    best_u = -1
    best_v = -1
    
    for u in range(n):
        for v in range(u + 1, n):
            if A[u][v] == 0:
                c4s = A3[u][v]
                if c4s < min_c4s:
                    min_c4s = c4s
                    best_u = u
                    best_v = v
    
    # Add the best edge
    if best_u != -1:
        A[best_u][best_v] = 1
        A[best_v][best_u] = 1
    else:
        # Should not happen for n=31, m=90
        # Pick any non-edge
        for u in range(n):
            for v in range(u + 1, n):
                if A[u][v] == 0:
                    A[u][v] = 1
                    A[v][u] = 1
                    best_u = u
                    best_v = v
                    break
            if best_u != -1:
                break
    
    # Extract edges list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 1:
                edges.append([i, j])
    
    # Verify edge count is 91
    assert len(edges) == 91, f"Edge count is {len(edges)}, expected 91"
    
    return edges
# EVOLVE-BLOCK-END
