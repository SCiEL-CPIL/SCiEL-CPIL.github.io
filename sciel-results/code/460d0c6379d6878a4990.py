"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP using O(n^2) per scan delta calculation.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute terms that depend on rows/cols of a and D
    # We will update D as we swap.
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Compute V_sum for all j > i
            # V_1 = a[i, :] @ D  (shape n)
            # V_2 = a.T[i, :] @ D (shape n)
            # C_1 = a @ D[:, i] (shape n)
            # C_2 = a.T @ D[:, i] (shape n)
            # C_3 = sum(a * D, axis=1) (shape n)
            # C_4 = sum(a.T * D, axis=1) (shape n)
            
            V_1 = a[i, :] @ D
            V_2 = a.T[i, :] @ D
            C_1 = a @ D[:, i]
            C_2 = a.T @ D[:, i]
            
            # C_3 and C_4 are constant for the current D, but D changes.
            # However, in one pass of i, D is fixed.
            # We can precompute C_3 and C_4 outside the i loop?
            # No, D changes when we swap.
            # But we only swap one pair per i iteration?
            # If we find a best swap and do it, D changes.
            # So C_3 and C_4 must be updated.
            
            # To keep it simple and correct, we recompute C_3, C_4 for each i?
            # That is O(n^2) per i, making total O(n^3).
            # Better: Update C_3, C_4 incrementally?
            # Or just compute them inside the loop. n=256, O(n^3) is ~16M ops.
            # In NumPy, O(n^3) with small constants might be okay for 120s?
            # 16M * 256 (iterations) = 4B ops. Too slow in Python/NumPy loop.
            
            # We need O(n^2) total per scan.
            # C_3[j] = sum_k a[j, k] D[j, k]
            # This is the diagonal of a * D summed? No.
            # It is the row-wise sum of element-wise product.
            
            # Let's try to avoid C_3, C_4 explicit computation if possible.
            # Or compute them once per scan and update?
            
            # Let's assume we compute them once per i.
            C_3 = np.sum(a * D, axis=1)
            C_4 = np.sum(a.T * D, axis=1)
            
            V_sum = V_1 + C_1 + V_2 + C_2 - C_3 - C_4
            
            # Compute T_i (constant for all j)
            # T_i = 2*a[i,i]*D[j,i] + 2*a[j,i]*D[i,i] - 2*a[j,i]*D[j,i]
            # Wait, T_i depends on j!
            # T_i(j) = 2*a[i,i]*D[j,i] + 2*a[j,i]*D[i,i] - 2*a[j,i]*D[j,i]
            
            # Compute T_j(j)
            # T_j(j) = a[i,j]*D[j,j] + a[j,i]*D[j,j] + 2*a[j,j]*D[i,j] - 2*a[j,j]*D[j,j]
            
            # Compute Delta_direct(j)
            # Delta_direct(j) = (a[i,i] - a[j,j])*(D[j,j] - D[i,i]) + (a[i,j] - a[j,i])*(D[j,i] - D[i,j])
            
            js = np.arange(i + 1, n)
            
            # Vectorize T_i, T_j, Delta_direct for js
            # D[:, i] is column i
            D_col_i = D[:, i]
            D_row_i = D[i, :]
            
            # T_i vector
            a_ii = a[i, i]
            D_ii = D[i, i]
            a_js_i = a[js, i]
            D_js_i = D[js, i]
            T_i_vec = 2 * a_ii * D_js_i + 2 * a_js_i * D_ii - 2 * a_js_i * D_js_i
            
            # T_j vector
            a_i_js = a[i, js]
            a_js_js = np.diag(a[js, :]) # a[j,j] for j in js
            D_js_js = np.diag(D[js, :]) # D[j,j] for j in js
            D_i_js = D[i, js]
            D_js_i_col = D[js, i] # D[j,i]
            
            T_j_vec = a_i_js * D_js_js + a_js_i * D_js_js + 2 * a_js_js * D_i_js - 2 * a_js_js * D_js_js
            # Simplify T_j_vec:
            # = (a_i_js + a_js_i) * D_js_js + 2 * a_js_js * (D_i_js - D_js_js)
            
            # Delta_direct vector
            Delta_direct = (a_ii - a_js_js) * (D_js_js - D_ii) + (a_i_js - a_js_i) * (D_js_i_col - D_i_js)
            
            # V_sum for js
            V_sum_js = V_sum[js]
            
            deltas = V_sum_js - T_i_vec - T_j_vec + Delta_direct
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                # Swap rows/cols i and j in D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Initial phase: Random restarts
    num_initials = 5
    time_per_initial = (time_budget * 0.3) / num_initials
    
    for _ in range(num_initials):
        if time.time() - start_time > time_budget * 0.3:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_per_initial, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # ILS phase
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        if time.time() - start_time > time_budget - 2:
            break
            
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
            break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
