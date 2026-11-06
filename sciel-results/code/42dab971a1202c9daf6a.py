import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n = 31
    q = 5
    
    # Generate points in PG(2, 5)
    points = []
    # Finite points (x, y, 1)
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity (x, 1, 0)
    for x in range(q):
        points.append((x, 1, 0))
    # Point (1, 0, 0)
    points.append((1, 0, 0))
    
    # Verify count
    assert len(points) == n, f"Expected 31 points, got {len(points)}"

    # Points as numpy array for faster dot product?
    # Or just list of tuples.
    
    best_c4_count = float('inf')
    best_edges = None
    best_form = None

    # Test different quadratic forms
    # (a, b, c) for a*x^2 + b*y^2 + c*z^2 = 0
    # We only care about the zero set, so scaling doesn't matter.
    # Distinct types over F5:
    # 1. (1, 1, 1)
    # 2. (1, 1, 2)
    # 3. (1, 2, 2)
    
    forms = [(1, 1, 1), (1, 1, 2), (1, 2, 2)]

    for a, b, c in forms:
        if time.time() - start_time > DEADLINE:
            break
            
        adj = np.zeros((n, n), dtype=np.int8)
        edges = []
        edge_set = set()

        for i in range(n):
            pi = points[i]
            for j in range(i + 1, n):
                pj = points[j]
                dot = (a * pi[0] * pj[0] + b * pi[1] * pj[1] + c * pi[2] * pj[2]) % q
                if dot == 0:
                    edges.append([i, j])
                    edge_set.add((i, j))
                    edge_set.add((j, i))
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Check if edge count is 90
        if len(edges) != 90:
            continue
            
        # Compute A^3
        # A is int8, A^3 might overflow? Max degree ~6. 6^3 = 216. Fits in int16.
        A3 = np.zeros((n, n), dtype=np.int16)
        # Use matrix multiplication
        # A @ A @ A
        A2 = adj.astype(np.int16) @ adj.astype(np.int16)
        A3 = A2 @ adj.astype(np.int16)
        
        # Find min A3[i,j] for non-edges
        min_val = n * n * n
        best_i, best_j = -1, -1
        
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    val = A3[i, j]
                    if val < min_val:
                        min_val = val
                        best_i, best_j = i, j
                        
        if min_val < best_c4_count:
            best_c4_count = min_val
            best_edges = edges.copy()
            best_form = (a, b, c)
            
    if best_edges is None:
        # Fallback to original logic if something went wrong
        # But forms should work
        pass

    # Add the best edge
    if best_i != -1 and best_j != -1:
        best_edges.append([best_i, best_j])
        
    return best_edges
