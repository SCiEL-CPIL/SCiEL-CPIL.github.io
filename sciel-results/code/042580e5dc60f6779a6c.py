import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10s safety margin, though it should take <0.1s
    
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
    
    assert len(points) == n, f"Expected 31 points, got {len(points)}"
    
    # Build adjacency matrix
    # A[i][j] = 1 if dot(points[i], points[j]) % q == 0 and i != j
    A = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                A[i][j] = 1
                A[j][i] = 1
    
    # Count edges in the base polarity graph
    edge_count = np.sum(A) // 2
    
    # The polarity graph of PG(2,5) should have 90 edges and be C4-free.
    # If edge_count is not 90, there might be an issue with the construction,
    # but for q=5, it is theoretically 90.
    if edge_count != 90:
        # If construction failed, we can't proceed optimally.
        # However, for this specific task, the construction is standard.
        pass
    
    # We need 91 edges.
    # We add one edge (u, v) that is not currently present.
    # The number of new C4s created is the number of 4-cycles involving (u,v).
    # A 4-cycle involving (u,v) is of the form u-w-v-z-u.
    # w and z must be common neighbors of u and v in the original graph.
    # The number of such pairs (w,z) is C(common_neighbors, 2).
    
    # Compute A^2 to get common neighbor counts
    # A2[i][j] = sum_k A[i][k] * A[k][j] = number of common neighbors of i and j
    A2 = A @ A
    
    # Find the pair (u, v) with u < v, A[u][v] == 0, and minimal C(A2[u][v], 2)
    min_c4s = float('inf')
    best_u = -1
    best_v = -1
    
    for u in range(n):
        for v in range(u + 1, n):
            if A[u][v] == 0:
                common = A2[u][v]
                c4s = common * (common - 1) // 2
                if c4s < min_c4s:
                    min_c4s = c4s
                    best_u = u
                    best_v = v
    
    # Add the best edge
    if best_u != -1:
        A[best_u][best_v] = 1
        A[best_v][best_u] = 1
    else:
        # Fallback: should not happen
        for u in range(n):
            for v in range(u + 1, n):
                if A[u][v] == 0:
                    A[u][v] = 1
                    A[v][u] = 1
                    break
            if A[u][v] == 1 and v > u: # crude check
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
