import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n = 16
    target_edges = 65
    
    # Precompute all possible triples (i, j, k) with i < j < k
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                
    # Precompute all possible edges (pairs)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    # Function to compute score (max degree sum over triangles)
    def compute_score(A, d):
        max_sum = 0
        for i, j, k in triples:
            if A[i, j] and A[j, k] and A[k, i]:
                s = d[i] + d[j] + d[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    # Initialize with baseline: K_{8,8} + edge (0,1)
    A = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            A[u, v] = 1
            A[v, u] = 1
    A[0, 1] = 1
    A[1, 0] = 1
    
    d = A.sum(axis=1)
    current_score = compute_score(A, d)
    
    best_A = A.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 5.0
    cool_rate = 0.999
    
    while time.time() - start_time < deadline:
        # Find a valid swap
        # Try a few times to find a valid swap
        valid_swap = False
        for _ in range(10):
            p1 = random.choice(pairs)
            p2 = random.choice(pairs)
            if A[p1[0], p1[1]] == 1 and A[p2[0], p2[1]] == 0:
                i, j = p1
                k, l = p2
                valid_swap = True
                break
        
        if not valid_swap:
            continue
            
        # Perform swap
        A[i, j] = 0
        A[j, i] = 0
        A[k, l] = 1
        A[l, k] = 1
        
        d[i] -= 1
        d[j] -= 1
        d[k] += 1
        d[l] += 1
        
        new_score = compute_score(A, d)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A.copy()
        else:
            # Reject, revert
            A[i, j] = 1
            A[j, i] = 1
            A[k, l] = 0
            A[l, k] = 0
            
            d[i] += 1
            d[j] += 1
            d[k] -= 1
            d[l] -= 1
            
        T *= cool_rate
        if T < 0.1:
            T = 0.1

    # Extract edges from best_A
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                edges_list.append([i, j])
                
    return {'n': n, 'edges': edges_list}
