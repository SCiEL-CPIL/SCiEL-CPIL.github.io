import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    w2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                w2.append(v)
                
    w3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    w3.append(v)
                    
    V = np.array(w2 + w3, dtype=np.int64)
    N = V.shape[0]
    
    # Compute conflict matrix
    # Dot products
    M = V @ V.T
    # Conflict if dot >= 2
    C = (M >= 2).astype(np.int64)
    np.fill_diagonal(C, 0)
    
    def greedy(C, mode='min'):
        remaining = set(range(N))
        solution = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_score = 0 if mode == 'min' else -1
            
            for idx in remaining:
                # Degree in remaining subgraph
                # Sum of C[idx, j] for j in remaining, j != idx
                # Since C is symmetric, we can sum row idx and mask by remaining
                deg = 0
                # Efficient way: 
                # We can maintain a list of neighbors?
                # Or just sum.
                # N is 1540. Summing 1540 elements is fast.
                deg = np.sum(C[idx, list(remaining)])
                
                if mode == 'min':
                    if best_idx == -1 or deg < best_score:
                        best_score = deg
                        best_idx = idx
                else:
                    if best_idx == -1 or deg > best_score:
                        best_score = deg
                        best_idx = idx
                        
            solution.append(idx)
            
            # Remove idx and neighbors
            neighbors = np.where(C[idx] > 0)[0]
            remaining.discard(idx)
            remaining.difference_update(neighbors)
            
        return [V[i] for i in solution]

    sol_min = greedy(C, 'min')
    sol_max = greedy(C, 'max')
    
    best_sol = sol_min if len(sol_min) > len(sol_max) else sol_max
    
    return np.array(best_sol, dtype=np.int64)
