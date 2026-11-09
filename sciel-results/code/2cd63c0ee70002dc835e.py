import numpy as np
import time
import random

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Initial graph: K_{8,8} + one edge
    # Vertices 0-7 in A, 8-15 in B
    A = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            A[u, v] = 1
            A[v, u] = 1
    # Add edge (0, 1)
    A[0, 1] = 1
    A[1, 0] = 1
    
    # Verify edge count
    current_edges = A.sum() // 2
    if current_edges != target_edges:
        # Fallback, though it should be 65
        pass

    def get_score(A):
        D = A.sum(axis=1)
        # Find triangles
        # L = A @ A
        # Triangles are i<j<k where A[i,j], A[j,k], A[i,k] are 1
        # We can just iterate over all pairs and check common neighbors
        # Or use the matrix multiplication trick
        L = A @ A
        # L[i,j] is number of common neighbors
        # If A[i,j] == 1 and L[i,j] > 0, then (i,j) is part of L[i,j] triangles
        # We need the max sum of degrees for any triangle.
        max_sum = 0
        # Iterate over edges
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    # Common neighbors
                    # common = np.where((A[i] == 1) & (A[j] == 1))[0]
                    # This is slow in Python loop.
                    # Let's use the L matrix.
                    # For each common neighbor k, triangle (i,j,k).
                    # Sum = D[i] + D[j] + D[k]
                    # We need max over k.
                    # So for edge (i,j), max triangle sum involving it is D[i] + D[j] + max(D[k] for k in common)
                    # But we need global max over all triangles.
                    # So we can just compute all triangle sums.
                    pass
        
        # Faster vectorized approach:
        # Create a list of triangles
        # Since n is small, we can just do a triple loop or use the L matrix
        # L[i,j] > 0 and A[i,j]==1 implies triangles.
        # For each such pair (i,j), the triangles are with all k such that A[i,k]==1 and A[j,k]==1.
        # The sum is D[i]+D[j]+D[k].
        # We want max over all i,j,k.
        
        # Let's just use a simple loop for clarity and correctness, n=16 is tiny.
        max_val = 0
        for k in range(n):
            # For each vertex k, look at its neighbors
            neighbors = np.where(A[k] == 1)[0]
            for idx_u in range(len(neighbors)):
                u = neighbors[idx_u]
                for idx_v in range(idx_u+1, len(neighbors)):
                    v = neighbors[idx_v]
                    if A[u, v] == 1:
                        # Triangle (k, u, v)
                        s = D[k] + D[u] + D[v]
                        if s > max_val:
                            max_val = s
        return max_val

    best_A = A.copy()
    best_score = get_score(best_A)
    current_A = A.copy()
    current_score = best_score
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        edges = np.argwhere(current_A == 1)
        edges = edges[edges[:, 0] < edges[:, 1]] # Unique edges
        if len(edges) == 0:
            break
        u, v = edges[random.randint(0, len(edges)-1)]
        
        # Pick a random non-edge to add
        non_edges = np.argwhere(current_A == 0)
        non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
        if len(non_edges) == 0:
            break
        x, y = non_edges[random.randint(0, len(non_edges)-1)]
        
        # Check validity: no self loops (already ensured by mask), 
        # and we are swapping an existing edge with a non-existing one.
        # Ensure (u,v) is not (x,y) or (y,x) - they are disjoint sets so this is fine.
        
        # Apply move
        current_A[u, v] = 0
        current_A[v, u] = 0
        current_A[x, y] = 1
        current_A[y, x] = 1
        
        new_score = get_score(current_A)
        
        # Acceptance
        if new_score < current_score or (T > T_min and random.random() < np.exp(-(new_score - current_score) / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A.copy()
        else:
            # Revert
            current_A[u, v] = 1
            current_A[v, u] = 1
            current_A[x, y] = 0
            current_A[y, x] = 0
            
        T *= cooling_rate
        
    # Convert best_A back to list of edges
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                best_edges.append([i, j])
                
    return {'n': n, 'edges': best_edges}
