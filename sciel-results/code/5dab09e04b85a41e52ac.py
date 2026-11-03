import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # Precompute M = A @ B^T
    M = a @ b.T
    diag_a = np.diag(a)
    
    # Initial permutation: heuristic based on row sums
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    idx_a = np.argsort(-a_sums)
    idx_b = np.argsort(-b_sums)
    p = np.zeros(n, dtype=np.int64)
    p[idx_a] = idx_b
    
    # Calculate initial cost
    def calc_cost(p):
        return np.einsum('ij,ij->', a, b[p, :])
        
    current_cost = calc_cost(p)
    
    # Local Search
    start_time = time.time()
    time_limit = 115  # Leave buffer
    
    b_T = b.T
    arange_n = np.arange(n)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Precompute for this pass
        K = a @ b[p, :].T
        X = np.einsum('jk,jk->j', a, b[p, :])
        
        # Iterate i
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
                
            u = p[i]
            C_i = a[i, :] @ b[u, :]
            Y = K[i, :]
            Z = M[:, u]
            
            dot_vals = Y - C_i - X + Z
            
            a_ij = a[i, :]
            a_ji = a[:, i]
            
            b_pj_j = b[p, arange_n]
            b_uj = b[u, :]
            
            b_pj_i = b[p, i]
            b_ui = b[u, i]
            
            T_j = (a_ij - diag_a) * (b_pj_j - b_uj)
            T_i = (a[i, i] - a_ji) * (b_pj_i - b_ui)
            
            deltas = 2 * (dot_vals - T_j - T_i)
            
            deltas[i] = np.inf
            
            best_j = np.argmin(deltas)
            
            if deltas[best_j] < 0:
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += deltas[best_j]
                improved = True
                break
        
        if not improved:
            break
            
    return p.tolist()
