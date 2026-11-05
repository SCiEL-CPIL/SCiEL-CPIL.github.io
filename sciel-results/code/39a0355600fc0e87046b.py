import numpy as np
import random
import time

def construct():
    n = 31
    edges_set = set()
    # Random initialization
    # Use a greedy random approach to start closer to optimal
    # Or just random 91 edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    # Start with a random set of 91 edges
    initial_edges = all_pairs[:91]
    A = np.zeros((n, n), dtype=int)
    for u, v in initial_edges:
        A[u, v] = 1
        A[v, u] = 1
        
    def count_c4(A):
        A2 = A @ A
        # Sum of binom(A2[u,v], 2) for u < v
        # binom(k, 2) = k*(k-1)/2
        # We need sum over u < v
        # A2 is symmetric
        # Use upper triangle
        upper = np.triu(A2, k=1)
        # Calculate k*(k-1)/2 for each element
        # Note: A2 elements are integers
        c4_count = 0.5 * np.sum(upper * (upper - 1))
        return c4_count

    current_c4 = count_c4(A)
    
    start_time = time.time()
    time_limit = 110  # Leave 10s for safety
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_swap = None
        
        # Get current edges and non-edges
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
        non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
        
        # To speed up, we can sample a subset of swaps if it's too slow
        # But 91 * 374 ~ 34k is fine.
        
        for u, v in edges_list:
            # Try removing (u, v)
            A[u, v] = 0
            A[v, u] = 0
            
            for x, y in non_edges_list:
                # Skip if x,y is u,v (impossible) or shares vertices?
                # If x,y shares vertex with u,v, it's a valid swap
                # Just add it
                A[x, y] = 1
                A[y, x] = 1
                
                new_c4 = count_c4(A)
                delta = new_c4 - current_c4
                
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (u, v, x, y)
                
                # Revert
                A[x, y] = 0
                A[y, x] = 0
            
            # Revert removal
            A[u, v] = 1
            A[v, u] = 1
            
        if best_swap:
            u, v, x, y = best_swap
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            current_c4 += best_delta
            improved = True
            # Optional: break early if we find a large improvement?
            # No, continue to find more.

    # Construct list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    return edges
