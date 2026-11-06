import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 40
    target_edges = 401
    
    # 1. Initialization: Turan Graph T2(40) + 1 edge
    # Partitions: 0-19, 20-39
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(20):
        for j in range(20, 40):
            A[i, j] = 1
            A[j, i] = 1
            
    # Add one edge within partition 0
    A[0, 1] = 1
    A[1, 0] = 1
    
    # List of edges for fast manipulation
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                edges.add((i, j))
                
    # Precompute all possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def calc_max_book(A):
        # B[i,j] = number of common neighbors
        # We only care about max over edges
        max_b = 0
        # Optimize: use matrix multiplication?
        # A @ A gives counts of length-2 paths.
        # (A @ A)[i,j] is common neighbors for i,j.
        # For edge (i,j), book size is (A @ A)[i,j].
        B = np.dot(A, A)
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    if B[i, j] > max_b:
                        max_b = B[i, j]
        return max_b

    current_max = calc_max_book(A)
    best_max = current_max
    best_A = A.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    # To speed up, we can cache B?
    # Recalculating B is O(n^3) = 64000 ops.
    # 10000 steps * 64000 = 640M ops. Might be too slow in Python.
    # Let's try to update B incrementally or just use a simpler heuristic.
    # Or, use a faster initialization and fewer, smarter steps.
    
    # Alternative: Greedy Local Search
    # Try to reduce the max book size by swapping edges.
    
    def get_edges_with_book(A, B):
        # Return list of (book_size, edge)
        res = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    res.append((B[i, j], (i, j)))
        return res

    B = np.dot(A, A)
    
    while time.time() < deadline:
        # Pick an edge with high book size to remove
        # Sort edges by book size
        edge_books = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    edge_books.append((B[i, j], i, j))
        
        # Sort descending
        edge_books.sort(reverse=True)
        
        # Pick top 10% candidates for removal
        num_cands = max(1, len(edge_books) // 10)
        candidates = edge_books[:num_cands]
        
        improved = False
        for _ in range(10): # Try 10 swaps per iteration
            if time.time() > deadline:
                break
                
            # Pick random candidate to remove
            _, u, v = random.choice(candidates)
            
            # Pick random non-edge to add
            # We need a non-edge (x, y)
            # Try random pairs
            found = False
            for _ in range(50):
                x, y = random.choice(all_pairs)
                if A[x, y] == 0:
                    found = True
                    break
            
            if not found:
                continue
            
            # Apply move
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            # Recalculate B (expensive but simple)
            # We can update B incrementally for the affected nodes
            # Nodes involved: u, v, x, y
            # But common neighbors of other pairs might change?
            # No, only pairs involving u, v, x, y or their neighbors?
            # Actually, B[i,j] changes if i or j is u, v, x, y OR if a common neighbor is removed/added.
            # This is complex. Let's just recalculate B if the move looks promising?
            # Or just recalculate max book for the affected edges?
            
            # Let's just recalculate the whole B.
            # np.dot is fast.
            B_new = np.dot(A, A)
            
            # Check max book
            new_max = 0
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1:
                        if B_new[i, j] > new_max:
                            new_max = B_new[i, j]
            
            if new_max < current_max:
                current_max = new_max
                B = B_new
                if current_max < best_max:
                    best_max = current_max
                    best_A = A.copy()
                improved = True
                # Re-calculate candidates?
                # Break inner loop to restart candidate selection
                break
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                # B remains unchanged (old B)
        
        if not improved and T > min_T:
            T *= cooling
            # If stuck, maybe perturb?
            # For now, just continue.
            
    # Return best_A edges
    edges_out = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges_out.append([i, j])
                
    return edges_out
