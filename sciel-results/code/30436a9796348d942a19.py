import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    for n in range(47, 38, -1):
        min_deg_required = n - 39
        if min_deg_required < 1:
            min_deg_required = 1
            
        for restart in range(100):
            if time.time() > deadline:
                break
                
            A = np.ones((n, n), dtype=np.int8)
            np.fill_diagonal(A, 0)
            
            # Randomize initial graph slightly
            # Remove 10% of edges randomly
            mask = np.random.rand(n, n) < 0.1
            A[mask & (A==1)] = 0
            A = np.maximum(A, A.T)
            np.fill_diagonal(A, 0)
            
            for _ in range(20000):
                if time.time() > deadline:
                    break
                    
                A2 = A @ A
                bad_mask = (A == 1) & (A2 >= 2)
                
                if not np.any(bad_mask):
                    degrees = np.sum(A, axis=1)
                    if np.min(degrees) >= min_deg_required:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if A[i,j] == 1:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        break
                else:
                    bad_indices = np.argwhere(bad_mask)
                    idx = random.randint(0, len(bad_indices)-1)
                    i, j = bad_indices[idx]
                    
                    common = np.where((A[i] == 1) & (A[j] == 1))[0]
                    if len(common) < 2:
                        continue
                        
                    # Pick two distinct common neighbors
                    # Shuffle common to pick random w, x
                    common_list = list(common)
                    random.shuffle(common_list)
                    w = common_list[0]
                    x = common_list[1]
                    
                    # Edges in C4 i-w-j-x-i
                    candidates = [(i, w), (w, j), (j, x), (x, i)]
                    
                    valid_candidates = []
                    for u, v in candidates:
                        if A[u, v] == 1:
                            if np.sum(A[u]) > min_deg_required and np.sum(A[v]) > min_deg_required:
                                valid_candidates.append((u, v))
                                
                    if not valid_candidates:
                        break
                        
                    u, v = random.choice(valid_candidates)
                    A[u, v] = 0
                    A[v, u] = 0
                    
        if time.time() > deadline:
            break
            
    return {'n': 39, 'edges': []}
