import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization using Hungarian Algorithm
    # Linearized cost: W[i, j] = sum_k a[i, k] * b[j, k]
    W = a @ b.T
    row_ind, col_ind = linear_sum_assignment(W)
    p = np.full(n, -1, dtype=np.int64)
    p[row_ind] = col_ind
    
    def get_cost(p):
        # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        # = sum_i a[i, :] . b[p[i], p]
        # We can compute this as trace(A @ B_p) where B_p is b with rows permuted?
        # Or simply sum of diagonal of A @ B_perm
        B_perm = b[:, p]
        S = a @ B_perm
        return int(np.sum(S[np.arange(n), p]))
        
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute a.T for column access if needed, but a[:, i] is fine.
    
    def compute_delta_swap(i, k, p):
        u, v = p[i], p[k]
        if u == v:
            return 0
        
        # Vector differences
        # b[v, p] is row v of b, indexed by p
        # b[u, p] is row u of b, indexed by p
        # We need:
        # 1. (a[i, :] - a[k, :]) dot (b[v, p] - b[u, p])
        # 2. (a[:, i] - a[:, k]) dot (b[p, v] - b[p, u])
        
        # To reduce overhead, we can compute:
        # row_diff = a[i, :] - a[k, :]
        # col_diff = a[:, i] - a[:, k]
        
        # But creating row_diff and col_diff is O(N).
        # Let's stick to the previous logic but ensure we use .dot() efficiently.
        
        # b[v, p] - b[u, p]
        diff_row = b[v, p] - b[u, p]
        # b[p, v] - b[p, u]
        diff_col = b[p, v] - b[p, u]
        
        # Term 1: (a[i, :] - a[k, :]) dot diff_row
        # = a[i, :] dot diff_row - a[k, :] dot diff_row
        t1 = np.dot(a[i, :], diff_row) - np.dot(a[k, :], diff_row)
        
        # Term 2: (a[:, i] - a[:, k]) dot diff_col
        # = a[:, i] dot diff_col - a[:, k] dot diff_col
        t2 = np.dot(a[:, i], diff_col) - np.dot(a[:, k], diff_col)
        
        total = t1 + t2
        
        # Corrections for j=i and j=k
        # The total above includes j=i and j=k.
        # For j=i:
        # term_i_val = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        # For j=k:
        # term_k_val = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        
        term_i_val = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        term_k_val = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        
        off_diag = total - term_i_val - term_k_val
        
        diag_ii = a[i,i] * (b[v,v] - b[u,u])
        diag_kk = a[k,k] * (b[u,u] - b[v,v])
        cross_ik = a[i,k] * (b[v,u] - b[u,v])
        cross_ki = a[k,i] * (b[u,v] - b[v,u])
        
        delta = diag_ii + diag_kk + cross_ik + cross_ki + off_diag
        return delta

    # Simulated Annealing
    # Estimate T
    sample_deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        d = compute_delta_swap(i, k, p)
        sample_deltas.append(abs(d))
    
    T = np.mean(sample_deltas) * 2.0 if sample_deltas else 10000.0
    T_min = 1.0
    alpha = 0.9999 # Slower cooling for more exploration
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for final polish
    
    while time.time() - start_time < time_limit:
        i, k = np.random.choice(n, 2, replace=False)
        delta = compute_delta_swap(i, k, p)
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            p[i], p[k] = p[k], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min

    # Final Local Search: Fast First-Improvement
    # Instead of O(N^2) full scan, do a limited number of random swaps that improve cost
    # Or a few passes of best-improvement on a subset.
    # Given time, let's do a quick local search using random pairs until no improvement found in X tries.
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    max_no_improve = 1000
    
    end_time = time.time()
    remaining_time = 118.0 - (end_time - start_time) # Total limit 120s, used ~110s in SA
    
    if remaining_time > 1.0:
        while no_improve_count < max_no_improve and time.time() - start_time < 118.0:
            i, k = np.random.choice(n, 2, replace=False)
            delta = compute_delta_swap(i, k, p)
            if delta < 0:
                p[i], p[k] = p[k], p[i]
                current_cost += delta
                no_improve_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                no_improve_count += 1
                
    return best_p.tolist()
