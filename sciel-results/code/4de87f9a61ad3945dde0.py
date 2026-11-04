"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with O(n^2) per scan.
    Precomputes C_3, C_4 once per scan and updates after swaps.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    a_diag = np.diag(a).copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Precompute C_3 and C_4 once per scan
        C_3 = np.sum(a * D, axis=1)
        C_4 = np.sum(a.T * D, axis=1)
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            V_1 = a[i, :] @ D
            V_2 = a.T[i, :] @ D
            C_1 = a @ D[:, i]
            C_2 = a.T @ D[:, i]
            
            V_sum = V_1 + C_1 + V_2 + C_2 - C_3 - C_4
            
            js = np.arange(i + 1, n)
            
            D_col_i = D[:, i]
            D_row_i = D[i, :]
            
            a_ii = a[i, i]
            D_ii = D[i, i]
            a_js_i = a[js, i]
            D_js_i = D[js, i]
            T_i_vec = 2 * a_ii * D_js_i + 2 * a_js_i * D_ii - 2 * a_js_i * D_js_i
            
            a_i_js = a[i, js]
            a_js_js = a_diag[js]
            D_js_js = np.diag(D[js, :])
            D_i_js = D[i, js]
            D_js_i_col = D[js, i]
            
            T_j_vec = (a_i_js + a_js_i) * D_js_js + 2 * a_js_js * (D_i_js - D_js_js)
            
            Delta_direct = (a_ii - a_js_js) * (D_js_js - D_ii) + (a_i_js - a_js_i) * (D_js_i_col - D_i_js)
            
            V_sum_js = V_sum[js]
            
            deltas = V_sum_js - T_i_vec - T_j_vec + Delta_direct
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                
                # Update D: swap rows and cols i and j
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                # Update C_3 and C_4 incrementally
                # C_3[k] = sum_m a[k,m] * D[k,m]
                # After swapping i and j in D, only rows i and j of D changed (and cols)
                # C_3[i] changes: was sum_m a[i,m]*D[i,m], now sum_m a[i,m]*D[j,m]
                # C_3[j] changes: was sum_m a[j,m]*D[j,m], now sum_m a[j,m]*D[i,m]
                # Other C_3[k] for k != i,j: D[k,i] and D[k,j] swapped, so
                # C_3[k] = sum_m a[k,m]*D[k,m] changes only if m=i or m=j
                # C_3[k] new = C_3[k] old - a[k,i]*D[k,i] - a[k,j]*D[k,j] + a[k,i]*D[k,j] + a[k,j]*D[k,i]
                
                # Save old values before swap already done above
                # We need old D[i,:] and D[j,:], D[:,i] and D[:,j]
                # But we already swapped. Let's recompute from current D.
                # Actually, let's just recompute C_3 and C_4 fully - it's O(n^2)
                # and we're doing O(n^2) per i anyway. But we can be smarter.
                
                # Recompute C_3 and C_4
                C_3 = np.sum(a * D, axis=1)
                C_4 = np.sum(a.T * D, axis=1)
                
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
    time_budget = 118  # leave 2s buffer for safety
    
    best_p = None
    best_cost = np.inf
    
    # Initial phase: Random restarts with greedy construction
    def _greedy_init():
        """Greedy construction: place facilities one by one at best location."""
        p = np.zeros(n, dtype=int)
        used = np.zeros(n, dtype=bool)
        # D[i,j] = b[p[i],p[j]] for placed, 0 for unplaced
        D = np.zeros((n, n))
        
        for i in range(n):
            # For each candidate location l, compute cost contribution
            best_l = -1
            best_c = np.inf
            for l in range(n):
                if used[l]:
                    continue
                # Cost = sum_j a[i,j]*b[l,p[j]] + sum_j a[j,i]*b[p[j],l]
                # + sum_{j,k} a[j,k]*b[p[j],p[k]] is constant
                c = a[i, :] @ D[:, l] + a[:, i] @ D[l, :]
                if c < best_c:
                    best_c = c
                    best_l = l
            p[i] = best_l
            used[best_l] = True
            # Update D: D[i, :] = b[best_l, p[:]], D[:, i] = b[p[:], best_l]
            D[i, :] = b[best_l, p]
            D[:, i] = b[p, best_l]
        return p
    
    num_initials = 4
    phase1_end = start_time + time_budget * 0.25
    
    # First initial: greedy
    p_init = _greedy_init()
    p_ls, cost_ls = _local_search(a, b, p_init, phase1_end, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Remaining initials: random
    for _ in range(num_initials - 1):
        if time.time() - start_time > phase1_end:
            break
        p_init = np.random.permutation(n)
        time_left = phase1_end - time.time()
        if time_left < 1:
            break
        p_ls, cost_ls = _local_search(a, b, p_init, time_left, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # ILS phase
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed - 5  # reserve 5s for final polish
    
    if remaining_time < 3:
        # Final polish
        p_final, cost_final = _local_search(a, b, best_p, time_budget - 1, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 5:
        if time.time() - start_time > time_budget - 5:
            break
            
        if consecutive_no_improve > 8:
            perturb_k = min(20, base_perturb + consecutive_no_improve)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        duration = (time_budget - 5) - (time.time() - start_time)
        if duration < 1:
            break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Final polish pass
    if time.time() - start_time < time_budget - 1:
        p_final, cost_final = _local_search(a, b, best_p, time_budget - 0.5, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
