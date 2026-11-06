import time
import random
import numpy as np

def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 120  # 120s safety margin
    
    # Initialize with K_{8,8} + edge (0,1)
    A = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            A[u, v] = 1
            A[v, u] = 1
    A[0, 1] = 1
    A[1, 0] = 1
    
    def get_score(A):
        deg = A.sum(axis=1)
        max_sum = 0
        # Iterate over all triples
        # Optimization: only check if A[i,j], A[j,k], A[i,k] are 1
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    for k in range(j+1, n):
                        if A[j, k] and A[i, k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = get_score(A)
    best_A = A.copy()
    best_score = current_score
    
    # Local Search
    # We will try to improve by swapping edges.
    # To speed up, we can precompute list of edges and non-edges.
    
    while time.time() < deadline:
        # Get current edges and non-edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        improved = False
        # Try all possible swaps
        # To avoid O(M^2) which is ~3500, and each swap requires score calc O(N^3) ~ 4000 ops,
        # total per iteration is ~ 14 million ops. In Python, this might be slow.
        # 14M ops in Python is ~ 1-2 seconds.
        # We can do ~ 60 iterations in 120s.
        
        # Let's optimize score calculation or swap evaluation.
        # Actually, let's just do a few random restarts and hill climbs.
        
        # Hill Climb Phase
        while time.time() < deadline:
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j]]
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not A[i, j]]
            
            best_swap = None
            best_new_score = current_score
            
            # Sample a subset of swaps to speed up if needed, but 3575 is small.
            for e1 in edges:
                for e2 in non_edges:
                    # Check if e1 and e2 are distinct (they are, one is edge, one is non-edge)
                    # Apply swap
                    A[e1[0], e1[1]] = 0
                    A[e1[1], e1[0]] = 0
                    A[e2[0], e2[1]] = 1
                    A[e2[1], e2[0]] = 1
                    
                    new_score = get_score(A)
                    
                    # Revert
                    A[e1[0], e1[1]] = 1
                    A[e1[1], e1[0]] = 1
                    A[e2[0], e2[1]] = 0
                    A[e2[1], e2[0]] = 0
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e1, e2)
            
            if best_swap:
                e1, e2 = best_swap
                A[e1[0], e1[1]] = 0
                A[e1[1], e1[0]] = 0
                A[e2[0], e2[1]] = 1
                A[e2[1], e2[0]] = 1
                current_score = best_new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = A.copy()
                improved = True
            else:
                # Local minimum
                break
        
        if time.time() >= deadline:
            break
            
        # Random Restart / Perturbation
        # Randomly swap a few edges
        for _ in range(5):
            e1 = random.choice(edges)
            e2 = random.choice(non_edges)
            A[e1[0], e1[1]] = 0
            A[e1[1], e1[0]] = 0
            A[e2[0], e2[1]] = 1
            A[e2[1], e2[0]] = 1
        current_score = get_score(A)

    # Return best found
    # Convert best_A to list of edges
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
