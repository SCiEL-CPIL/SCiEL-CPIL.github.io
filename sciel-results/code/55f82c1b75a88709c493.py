import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 150s kill

    # 1. Generate 31 points of PG(2,5)
    q = 5
    points = []
    points.append((1, 0, 0))
    points.append((0, 1, 0))
    points.append((0, 0, 1))
    for y in range(1, q):
        points.append((1, y, 0))
    for z in range(1, q):
        points.append((1, 0, z))
    for z in range(1, q):
        points.append((0, 1, z))
    for y in range(1, q):
        for z in range(1, q):
            points.append((1, y, z))
    
    n = len(points)
    
    # 2. Define Bilinear Form B(P, Q) = P[0]Q[0] + P[1]Q[1] + P[2]Q[2] mod 5
    # Note: The current program's bilinear form creates a graph with 65 edges.
    # We need 91 edges.
    
    # 3. Construct Base Graph (Polarity Graph)
    # We will use a NumPy matrix for faster operations.
    A = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for i in range(n):
        for j in range(i+1, n):
            p_i = points[i]
            p_j = points[j]
            val = (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append([i, j])
                
    base_edge_count = len(edges)
    target_edges = 91
    edges_to_add = target_edges - base_edge_count
    
    # Identify non-edges
    # We want to add edges that create the fewest 4-cycles.
    # Adding edge (u, v) creates C4s equal to the number of common neighbors of u and v.
    # Let CN(u, v) = |N(u) ∩ N(v)|.
    # We want to greedily pick the edge (u, v) with minimum CN(u, v).
    
    # Precompute common neighbors counts for all pairs?
    # n=31, pairs=465. 31^3 = 29791 ops. Very fast.
    # But A changes as we add edges. So we must recompute or update.
    
    # Helper to count common neighbors using vectorized ops
    def count_common_neighbors(u, v, A_mat):
        # A_mat[u] and A_mat[v] are vectors
        return np.sum(A_mat[u] * A_mat[v])
    
    added_edges = []
    
    # To optimize the search for the best edge, we can pre-sort potential edges
    # based on their initial common neighbor count, then update as we go?
    # Or just re-scan all non-edges each time. 
    # 26 edges to add. 465 non-edges. 26 * 465 * 31 ops ~ 375k ops. Negligible.
    
    # However, the "inspiration" was just lexicographic first 91.
    # The current program does greedy.
    # Let's stick to the greedy approach but ensure it's robust and fast.
    
    # List of candidate edges (non-edges)
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 0:
                non_edges.append((i, j))
    
    # Sort non-edges by initial common neighbor count to guide the search?
    # Greedy choice: at each step, pick the non-edge with min current common neighbors.
    
    for _ in range(edges_to_add):
        if time.time() > deadline:
            break
            
        best_edge = None
        min_count = float('inf')
        
        for (i, j) in non_edges:
            if A[i, j] == 1:
                continue
            c = count_common_neighbors(i, j, A)
            if c < min_count:
                min_count = c
                best_edge = (i, j)
                
        if best_edge is None:
            break
            
        i, j = best_edge
        A[i, j] = 1
        A[j, i] = 1
        edges.append([i, j])
        
    return edges

