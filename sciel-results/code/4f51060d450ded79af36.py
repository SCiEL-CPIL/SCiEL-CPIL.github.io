"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Precompute
    a_sum_row = a.sum(axis=1)
    b_sum_row = b.sum(axis=1)
    
    def compute_cost(p):
        # p is a permutation array: facility i goes to location p[i]
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        ap = p  # location assigned to each facility
        # Vectorized: sum_i sum_j a[i,j] * b[ap[i], ap[j]]
        return np.sum(a * b[np.ix_(ap, ap)])
    
    def cost_delta_swap(p, i, j):
        """Compute cost change when swapping facilities i and j (locations p[i] and p[j])."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        # Current contribution involving i and j
        # After swap: facility i goes to lj, facility j goes to li
        # delta = sum_k a[i,k]*(b[lj,k] - b[li,k]) + sum_k a[k,i]*(b[k,lj] - b[k,li])
        #       + sum_k a[j,k]*(b[li,k] - b[lj,k]) + sum_k a[k,j]*(b[k,li] - b[k,lj])
        #       + a[i,j]*(b[li,lj] - b[li,li]) + a[j,i]*(b[lj,li] - b[lj,lj])
        # Simplified:
        row_i = a[i, :]
        col_i = a[:, i]
        row_j = a[j, :]
        col_j = a[:, j]
        
        b_li = b[li, :]
        b_lj = b[lj, :]
        b_col_li = b[:, li]
        b_col_lj = b[:, lj]
        
        delta = (row_i @ (b_lj - b_li) 
                 + col_i @ (b_col_lj - b_col_li)
                 + row_j @ (b_li - b_lj)
                 + col_j @ (b_col_li - b_col_lj)
                 + a[i, j] * (b[li, lj] - b[li, li])
                 + a[j, i] * (b[lj, li] - b[lj, lj]))
        return delta
    
    # Initial solution: rank matching
    heavy = np.argsort(-a_sum_row, kind='stable')
    central = np.argsort(b_sum_row, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Simulated Annealing
    T_start = best_cost * 0.05 if best_cost > 0 else 1000.0
    T_end = max(0.5, best_cost * 0.0001)
    cooling = 0.9995
    
    T = T_start
    iterations = 0
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and T > T_end:
        # Randomly pick two facilities to swap
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        
        delta = cost_delta_swap(p, i, j)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            # Accept the move
            p[i], p[j] = p[j], p[i]
            cur_cost = best_cost + delta if delta < 0 else best_cost  # approximate tracking
            # Actually track cost properly
            if delta < 0:
                best_cost += delta
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            iterations += 1
        else:
            no_improve += 1
            iterations += 1
        
        T *= cooling
        
        # Check deadline periodically
        if iterations % 10000 == 0 and time.time() > deadline:
            break
        
        if no_improve > max_no_improve and T < T_start * 0.1:
            # Try a restart with perturbation
            if time.time() > deadline - 30:
                break
            # Perturb best solution
            p = best_p.copy()
            for _ in range(n // 5):
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                p[i], p[j] = p[j], p[i]
            best_cost = compute_cost(p)
            T = max(T, T_start * 0.05)
            no_improve = 0
            max_no_improve = 30000
    
    # Final local polish: 2-opt pass on best solution
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    break
                delta = cost_delta_swap(best_p, i, j)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
