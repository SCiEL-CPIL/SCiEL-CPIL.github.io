import numpy as np
import random
import time

def construct():
    n = 16
    # Initial construction: K_{6,10} + 5 disjoint edges in part B
    # Part A: 0-5, Part B: 6-15
    A = np.zeros((n, n), dtype=np.uint8)
    
    # K_{6,10}
    for u in range(6):
        for v in range(6, 16):
            A[u, v] = 1
            A[v, u] = 1
            
    # 5 disjoint edges in B
    edges_b = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
    for u, v in edges_b:
        A[u, v] = 1
        A[v, u] = 1
        
    # Precompute all triplets as indices
    triplets = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triplets.append((i, j, k))
                
    def get_score(A):
        deg = A.sum(axis=1)
        max_sum = 0
        # Local variables for speed
        A_local = A
        deg_local = deg
        for i, j, k in triplets:
            if A_local[i, j] and A_local[j, k] and A_local[i, k]:
                s = deg_local[i] + deg_local[j] + deg_local[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    best_A = A.copy()
    best_score = get_score(best_A)
    current_A = A.copy()
    current_score = best_score
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    T = 1.0
    cooling = 0.9995
    
    # To speed up random edge selection, we can maintain a list of edges
    # But for N=16, random picking is fine.
    
    while time.time() < deadline:
        # Pick random edge to remove
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u == v or current_A[u, v] == 0:
            continue
            
        # Pick random edge to add
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        if x == y or current_A[x, y] == 1:
            continue
            
        # Ensure (u,v) and (x,y) are distinct
        if {u, v} == {x, y}:
            continue
            
        # Apply move
        current_A[u, v] = 0
        current_A[v, u] = 0
        current_A[x, y] = 1
        current_A[y, x] = 1
        
        new_score = get_score(current_A)
        
        # Acceptance
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
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
            
        T *= cooling
        
    # Reconstruct edges list from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
