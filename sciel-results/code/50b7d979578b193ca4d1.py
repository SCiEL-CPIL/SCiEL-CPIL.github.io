import numpy as np
import time
import random

def _count_violations_numpy(adj_matrices):
    total_violations = 0
    for A in adj_matrices:
        common = A @ A.T
        mask = common >= 2
        np.fill_diagonal(mask, False)
        n = A.shape[0]
        i, j = np.triu_indices(n, k=1)
        total_violations += np.sum(mask[i, j])
    return total_violations

def construct():
    deadline = time.time() + 140
    random.seed(42)
    np.random.seed(42)
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try n = 27, 26, 25
    for target_n in [27, 26, 25]:
        if time.time() > deadline - 30:
            break
            
        num_edges = target_n * (target_n - 1) // 2
        edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
        found = False
        
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            adj = [np.zeros((target_n, target_n), dtype=np.int8) for _ in range(5)]
            colors = np.zeros(num_edges, dtype=np.int8)
            
            for idx, (i, j) in enumerate(edges):
                c = np.random.randint(0, 5)
                colors[idx] = c
                adj[c][i, j] = 1
                adj[c][j, i] = 1
            
            current_violations = _count_violations_numpy(adj)
            
            T = 1.0
            min_T = 0.05
            cooling = 0.99995
            
            for it in range(200000):
                if time.time() > deadline - 5:
                    break
                
                if current_violations == 0:
                    found = True
                    break
                
                idx = np.random.randint(0, num_edges)
                i, j = edges[idx]
                old_c = colors[idx]
                new_c = np.random.randint(0, 5)
                if new_c == old_c:
                    continue
                
                adj[old_c][i, j] = 0
                adj[old_c][j, i] = 0
                adj[new_c][i, j] = 1
                adj[new_c][j, i] = 1
                colors[idx] = new_c
                
                new_violations = _count_violations_numpy(adj)
                delta = new_violations - current_violations
                
                if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                    current_violations = new_violations
                else:
                    adj[new_c][i, j] = 0
                    adj[new_c][j, i] = 0
                    adj[old_c][i, j] = 1
                    adj[old_c][j, i] = 1
                    colors[idx] = old_c
                
                T *= cooling
                if T < min_T:
                    T = min_T
            
            if found:
                best_n = target_n
                best_colors = colors.tolist()
                break
        
        if found:
            break

    return {'n': best_n, 'colors': best_colors}
