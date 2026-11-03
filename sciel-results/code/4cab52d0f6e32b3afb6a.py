"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def fast_delta_swap(p, i, j):
        pi = p[i]
        pj = p[j]
        
        # Old Cost S
        # Row i: sum_v A[i,v] B[pi, p[v]]
        old_row_i = np.dot(a[i, :], b[pi, p])
        # Col i: sum_u A[u,i] B[p[u], pi]
        old_col_i = np.dot(a[:, i], b[p, pi])
        # Row j: sum_v A[j,v] B[pj, p[v]]
        old_row_j = np.dot(a[j, :], b[pj, p])
        # Col j: sum_u A[u,j] B[p[u], pj]
        old_col_j = np.dot(a[:, j], b[p, pj])
        
        # Subtract double counted terms in Old
        old_ii = a[i, i] * b[pi, pi]
        old_ij = a[i, j] * b[pi, pj]
        old_ji = a[j, i] * b[pj, pi]
        old_jj = a[j, j] * b[pj, pj]
        
        old_cost_s = old_row_i + old_col_i + old_row_j + old_col_j - old_ii - old_ij - old_ji - old_jj
        
        # New Cost S
        # p_new[i] = pj, p_new[j] = pi
        
        # Row i (now at pj): sum_v A[i,v] B[pj, p_new[v]]
        # Base: b[pj, p]
        # Adjust i: b[pj, pj] - b[pj, pi]
        # Adjust j: b[pj, pi] - b[pj, pj]
        new_row_i = np.dot(a[i, :], b[pj, p]) + a[i, i]*(b[pj, pj] - b[pj, pi]) + a[i, j]*(b[pj, pi] - b[pj, pj])
        
        # Col i (now at pj): sum_u A[u,i] B[p_new[u], pj]
        # Base: b[p, pj]
        # Adjust i: b[pj, pj] - b[pi, pj]
        # Adjust j: b[pi, pj] - b[pj, pj]
        new_col_i = np.dot(a[:, i], b[p, pj]) + a[i, i]*(b[pj, pj] - b[pi, pj]) + a[j, i]*(b[pi, pj] - b[pj, pj])
        
        # Row j (now at pi): sum_v A[j,v] B[pi, p_new[v]]
        # Base: b[pi, p]
        # Adjust i: b[pi, pj] - b[pi, pi]
        # Adjust j: b[pi, pi] - b[pi, pj]
        new_row_j = np.dot(a[j, :], b[pi, p]) + a[j, i]*(b[pi, pj] - b[pi, pi]) + a[j, j]*(b[pi, pi] - b[pi, pj])
        
        # Col j (now at pi): sum_u A[u,j] B[p_new[u], pi]
        # Base: b[p, pi]
        # Adjust i: b[pj, pi] - b[pi, pi]
        # Adjust j: b[pi, pi] - b[pj, pi]
        new_col_j = np.dot(a[:, j], b[p, pi]) + a[i, j]*(b[pj, pi] - b[pi, pi]) + a[j, j]*(b[pi, pi] - b[pj, pi])
        
        # Subtract double counted terms in New
        new_ii = a[i, i] * b[pj, pj]
        new_ij = a[i, j] * b[pj, pi]
        new_ji = a[j, i] * b[pi, pj]
        new_jj = a[j, j] * b[pi, pi]
        
        new_cost_s = new_row_i + new_col_i + new_row_j + new_col_j - new_ii - new_ij - new_ji - new_jj
        
        return new_cost_s - old_cost_s

    def local_search(p, current_cost):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    delta = fast_delta_swap(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    # Initializations
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # 1. Hungarian Proxy
    proxy_cost = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(proxy_cost)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    # 2. Random
    p_rand = np.random.permutation(n).astype(np.int64)
    
    # 3. Greedy Sort
    # Sort facilities by descending row sum
    # Sort locations by ascending row sum
    fac_order = np.argsort(-a_row_sums)
    loc_order = np.argsort(b_row_sums)
    p_greedy = np.empty(n, dtype=np.int64)
    for k in range(n):
        p_greedy[fac_order[k]] = loc_order[k]
    
    start_time = time.time()
    time_limit = 118.0
    slice_time = time_limit / 3.0
    
    best_p = None
    best_cost = float('inf')
    
    def run_sa(start_p, time_slice):
        p = start_p.copy()
        current_cost = calc_cost(p)
        best_p_local = p.copy()
        best_cost_local = current_cost
        
        T = 1000.0
        cooling_rate = 0.9999
        # Adjust cooling rate based on time slice? 
        # If time_slice is short, cool slower to explore? Or faster to converge?
        # Let's stick to standard.
        
        t_start = time.time()
        while time.time() - t_start < time_slice:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            delta = fast_delta_swap(p, i, j)
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost_local:
                    best_cost_local = current_cost
                    best_p_local = p.copy()
            
            T *= cooling_rate
            if T < 1.0:
                T = 1.0 # Floor temperature
        
        return best_p_local, best_cost_local

    # Run Slice 1
    p1, c1 = run_sa(p_hung, slice_time)
    p1, c1 = local_search(p1, c1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1

    # Run Slice 2
    p2, c2 = run_sa(p_rand, slice_time)
    p2, c2 = local_search(p2, c2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2

    # Run Slice 3
    p3, c3 = run_sa(p_greedy, slice_time)
    p3, c3 = local_search(p3, c3)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3

    return best_p.tolist()

# EVOLVE-BLOCK-END
