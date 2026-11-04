import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Ensure we're working with int64
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost_full(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    def compute_dynamic_cost(p):
        """
        Compute the linear cost matrix for the QAP given current assignment p.
        C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        """
        b_jp = b[:, p]  # b_jp[j, k] = b[j, p[k]]
        b_pj = b[p, :]  # b_pj[k, j] = b[p[k], j]
        C_first = a @ b_jp.T   # C_first[i, j] = sum_k a[i,k] * b_jp.T[k,j] = sum_k a[i,k]*b[j,p[k]]
        C_second = a.T @ b_pj  # C_second[i, j] = sum_k a[k,i] * b_pj[k,j] = sum_k a[k,i]*b[p[k],j]
        return C_first + C_second
    
    # Initial assignment: Rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    current_cost = calc_cost_full(p)
    best_p = p.copy()
    best_cost = current_cost
    
    def local_search_2swap(p, cost, time_budget=5.0):
        """
        Efficient 2-swap local search.
        """
        p = p.copy()
        cost = int(cost)
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            b_p = b[np.ix_(p, p)]  # b_p[i,j] = b[p[i],p[j]]
            
            for i in range(n):
                if time.time() - start > time_budget:
                    return p, cost
                pi = p[i]
                
                # Precompute vectors for facility i
                # v1[j] = sum_k a[i,k] * b[p[j], p[k]]
                v1 = a[i, :] @ b_p.T
                # v2[j] = sum_k a[k,i] * b[p[k], p[j]]
                v2 = a[:, i] @ b_p  # b_p[k,j] = b[p[k],p[j]], so a[:,i] @ b_p gives sum_k a[k,i]*b[p[k],p[j]]
                
                # For the j-related terms, we need:
                # sum_k a[j,k]*(b[pi,pk] - b[pj,pk])
                # = sum_k a[j,k]*b[pi,p[k]] - sum_k a[j,k]*b[p[j],p[k]]
                # = (a[j,:] @ b[pi, p]) - (a[j,:] @ b[p[j], p])
                # Let's precompute b_pi_p = b[pi, p] (1D array of length n)
                # and for each j, b_pj_p = b[p[j], p]
                
                b_pi_p = b[pi, p]  # b_pi_p[k] = b[pi, p[k]]
                
                # For column part of j:
                # sum_k a[k,j]*(b[pk,pi] - b[pk,pj])
                # = sum_k a[k,j]*b[p[k],pi] - sum_k a[k,j]*b[p[k],p[j]]
                # = (a[:,j] @ b[p, pi]) - (a[:,j] @ b[p, p[j]])
                # Let b_p_pi = b[p, pi] (1D array)
                b_p_pi = b[p, pi]  # b_p_pi[k] = b[p[k], pi]
                
                for j in range(i + 1, n):
                    if time.time() - start > time_budget:
                        return p, cost
                    pj = p[j]
                    
                    # Compute delta for swapping p[i] and p[j]
                    # The cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
                    # When we swap positions pi and pj (facilities i and j), 
                    # the terms that change are those involving i or j.
                    
                    # Let's compute the delta directly.
                    # Old cost contribution involving i or j:
                    # sum_k a[i,k]*b[pi,p[k]] + sum_k a[k,i]*b[p[k],pi]
                    # + sum_k a[j,k]*b[pj,p[k]] + sum_k a[k,j]*b[p[k],pj]
                    # - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]  (to avoid double counting i,j pair)
                    # Actually, let's just compute the full delta using the formula:
                    # delta = sum_k a[i,k]*(b[pj,pk] - b[pi,pk]) 
                    #       + sum_k a[k,i]*(b[pk,pj] - b[pk,pi])
                    #       + sum_k a[j,k]*(b[pi,pk] - b[pj,pk])
                    #       + sum_k a[k,j]*(b[pk,pi] - b[pk,pj])
                    #       + a[i,j]*(b[pj,pi] - b[pi,pj])
                    #       + a[j,i]*(b[pi,pj] - b[pj,pi])
                    #       + a[i,i]*(b[pj,pj] - b[pi,pi])
                    #       + a[j,j]*(b[pi,pi] - b[pj,pj])
                    
                    # Using precomputed vectors:
                    # sum_k a[i,k]*(b[pj,pk] - b[pi,pk]) = v1[j] - v1[i]
                    # sum_k a[k,i]*(b[pk,pj] - b[pk,pi]) = v2[j] - v2[i]
                    
                    # For j terms:
                    # sum_k a[j,k]*(b[pi,pk] - b[pj,pk])
                    # = sum_k a[j,k]*b[pi,p[k]] - sum_k a[j,k]*b[p[j],p[k]]
                    # = (a[j,:] @ b_pi_p) - (a[j,:] @ b[p[j], p])
                    # Note: a[j,:] @ b[p[j],p] = sum_k a[j,k]*b[p[j],p[k]]
                    
                    # sum_k a[k,j]*(b[pk,pi] - b[pk,pj])
                    # = sum_k a[k,j]*b[p[k],pi] - sum_k a[k,j]*b[p[k],p[j]]
                    # = (a[:,j] @ b_p_pi) - (a[:,j] @ b[p, p[j]])
                    
                    # This is getting complex. Let's just compute it directly for each (i,j)
                    # but use numpy operations for the inner sums.
                    
                    # mask approach
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    pk = p[mask]
                    
                    delta = 0.0
                    # Self terms
                    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
                    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
                    # Cross i,j
                    delta += a[i, j] * (b[pj, pi] - b[pi, pj])
                    delta += a[j, i] * (b[pi, pj] - b[pj, pi])
                    # Bulk terms
                    delta += np.sum(a[i, mask] * (b[pj, pk] - b[pi, pk]))
                    delta += np.sum(a[mask, i] * (b[pk, pj] - b[pk, pi]))
                    delta += np.sum(a[j, mask] * (b[pi, pk] - b[pj, pk]))
                    delta += np.sum(a[mask, j] * (b[pk, pi] - b[pk, pj]))
                    
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += int(round(delta))
                        improved = True
                        # Need to recompute b_p since p changed
                        b_p = b[np.ix_(p, p)]
                        break  # Restart the j loop for this i since p changed
        
        return p, cost
    
    # Main loop
    T = 5000.0
    min_T = 10.0
    
    # Initial local search
    p, current_cost = local_search_2swap(p, current_cost, time_budget=10.0)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    while time.time() < deadline:
        # Compute dynamic cost matrix
        C = compute_dynamic_cost(p)
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        delta = new_cost - current_cost
        if delta < 0 or (T > min_T and np.random.rand() < np.exp(-delta / T)):
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = new_p.copy()
            
            p = new_p.copy()
            current_cost = new_cost
            
            # Local search refinement
            p, current_cost = local_search_2swap(p, current_cost, time_budget=3.0)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            T = max(min_T, T * 0.9)
        else:
            # Random 2-swap perturbation
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                old_i, old_j = p[i], p[j]
                p[i], p[j] = old_j, old_i
                swap_cost = calc_cost_full(p)
                swap_delta = swap_cost - current_cost
                if swap_delta < 0 or (T > min_T and np.random.rand() < np.exp(-swap_delta / T)):
                    if swap_cost < best_cost:
                        best_cost = swap_cost
                        best_p = p.copy()
                    current_cost = swap_cost
                    T = max(min_T, T * 0.9)
                else:
                    p[i], p[j] = old_i, old_j
                    T = max(min_T, T * 0.98)
    
    return best_p.tolist()
