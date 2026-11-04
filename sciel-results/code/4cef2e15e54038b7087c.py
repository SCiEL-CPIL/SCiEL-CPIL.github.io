"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def delta_swap(p, i, j):
        pi = p[i]
        pj = p[j]
        
        # Efficient delta calculation for swap of i and j
        # We only need to consider terms involving i or j
        
        # Old contributions:
        # Terms where i is row: sum_k a[i,k] b[pi, p[k]]
        # Terms where i is col: sum_k a[k,i] b[p[k], pi]
        # Terms where j is row: sum_k a[j,k] b[pj, p[k]]
        # Terms where j is col: sum_k a[k,j] b[p[k], pj]
        
        # Note: (i,i), (i,j), (j,i), (j,j) are counted in multiple terms above.
        # Specifically:
        # (i,i) is in "i is row" (k=i) and "i is col" (k=i)
        # (i,j) is in "i is row" (k=j) and "j is col" (k=i)
        # (j,i) is in "j is row" (k=i) and "i is col" (k=j)
        # (j,j) is in "j is row" (k=j) and "j is col" (k=j)
        
        # To avoid double counting issues, let's compute the total change directly.
        
        # Let's define the cost contribution of a set of indices S to the total cost.
        # When we swap i and j, only terms where at least one index is i or j change.
        
        # Let's compute old and new costs for the "affected" terms.
        # Affected terms are all (u,v) such that u in {i,j} or v in {i,j}.
        
        # This set is:
        # (i,i), (i,j), (i,k) for k not in {i,j}
        # (j,i), (j,j), (j,k) for k not in {i,j}
        # (k,i), (k,j) for k not in {i,j}
        
        # Total affected = Row i + Col i + Row j + Col j - (i,i) - (i,j) - (j,i) - (j,j)
        # Wait, Row i = {(i,v) for all v}, Col i = {(u,i) for all u}
        # Row i + Col i includes (i,i) twice, (i,j) once, (j,i) once, (i,k) once, (k,i) once.
        # Row j + Col j includes (j,j) twice, (j,i) once, (i,j) once, (j,k) once, (k,j) once.
        # So Row i + Col i + Row j + Col j includes:
        # (i,i): 2 times
        # (i,j): 1 (from Row i) + 1 (from Col j) = 2 times
        # (j,i): 1 (from Row j) + 1 (from Col i) = 2 times
        # (j,j): 2 times
        # (i,k), (k,i), (j,k), (k,j): 1 time each
        
        # So if we sum Row i + Col i + Row j + Col j, we count (i,i), (i,j), (j,i), (j,j) twice.
        # We need to subtract them once to get the correct count.
        
        # Old Cost of Affected:
        old_row_i = np.dot(a[i, :], b[pi, p])
        old_col_i = np.dot(a[:, i], b[p, pi])
        old_row_j = np.dot(a[j, :], b[pj, p])
        old_col_j = np.dot(a[:, j], b[p, pj])
        
        # Subtract double-counted diagonal/cross terms
        old_diag_corr = a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
        
        old_affected = old_row_i + old_col_i + old_row_j + old_col_j - old_diag_corr
        
        # New Cost of Affected:
        # p_new[i] = pj, p_new[j] = pi
        
        # New Row i: sum_v a[i,v] b[pj, p_new[v]]
        # p_new[v] = pj if v==i, pi if v==j, p[v] otherwise
        # So b[pj, p_new] is a vector where:
        # index i: b[pj, pj]
        # index j: b[pj, pi]
        # index k: b[pj, p[k]]
        
        new_row_i_vec = b[pj, p].copy()
        new_row_i_vec[i] = b[pj, pj]
        new_row_i_vec[j] = b[pj, pi]
        new_row_i = np.dot(a[i, :], new_row_i_vec)
        
        # New Col i: sum_u a[u,i] b[p_new[u], pj]
        # p_new[u] = pj if u==i, pi if u==j, p[u] otherwise
        # So b[p_new, pj] is a vector where:
        # index i: b[pj, pj]
        # index j: b[pi, pj]
        # index k: b[p[k], pj]
        
        new_col_i_vec = b[p, pj].copy()
        new_col_i_vec[i] = b[pj, pj]
        new_col_i_vec[j] = b[pi, pj]
        new_col_i = np.dot(a[:, i], new_col_i_vec)
        
        # New Row j: sum_v a[j,v] b[pi, p_new[v]]
        # p_new[v] = pj if v==i, pi if v==j, p[v] otherwise
        # So b[pi, p_new] is a vector where:
        # index i: b[pi, pj]
        # index j: b[pi, pi]
        # index k: b[pi, p[k]]
        
        new_row_j_vec = b[pi, p].copy()
        new_row_j_vec[i] = b[pi, pj]
        new_row_j_vec[j] = b[pi, pi]
        new_row_j = np.dot(a[j, :], new_row_j_vec)
        
        # New Col j: sum_u a[u,j] b[p_new[u], pi]
        # p_new[u] = pj if u==i, pi if u==j, p[u] otherwise
        # So b[p_new, pi] is a vector where:
        # index i: b[pj, pi]
        # index j: b[pi, pi]
        # index k: b[p[k], pi]
        
        new_col_j_vec = b[p, pi].copy()
        new_col_j_vec[i] = b[pj, pi]
        new_col_j_vec[j] = b[pi, pi]
        new_col_j = np.dot(a[:, j], new_col_j_vec)
        
        # New diag corr
        new_diag_corr = a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
        
        new_affected = new_row_i + new_col_i + new_row_j + new_col_j - new_diag_corr
        
        return new_affected - old_affected
    
    # 1. Initialization: Hungarian-based rank structure
    # We use a proxy cost matrix to guide the initial assignment.
    # We want to match high flow facilities with low distance locations.
    # proxy_cost[i, j] = a_row_sums[i] * b_row_sums[j]
    # Minimizing this encourages high a_rs to be with low b_rs.
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    proxy_cost = np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(proxy_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Simulated Annealing with Restart
    # We will run multiple SA phases, restarting from best solution if stuck.
    
    T_start = 1000.0
    T_end = 1.0
    cooling_rate = 0.99995
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        T = T_start
        no_improve_counter = 0
        
        while T > T_end and time.time() - start_time < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            delta = delta_swap(p, i, j)
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            
            T *= cooling_rate
            
            if no_improve_counter > max_no_improve:
                break
        
        # If we broke out due to no improvement, restart from best
        if no_improve_counter > max_no_improve and time.time() - start_time < time_limit:
            p = best_p.copy()
            current_cost = best_cost
    
    # 3. Final Local Search Polish
    # Try all pairs for improvement.
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = delta_swap(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
