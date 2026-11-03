import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)
    
    # Initial assignment: Rank matching
    # Heavy flows (high row sum in A) to central locations (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum
    heavy_order = np.argsort(-a_sums, kind='stable')
    # Sort locations by increasing distance sum (central)
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Calculate initial cost
    def calc_cost(p):
        # b_p[i, j] = b[p[i], p[j]]
        # cost = sum(a[i, j] * b[p[i], p[j]])
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Hybrid approach:
    # 1. Use Hungarian-based linearization to make large jumps (escape local optima)
    # 2. Use Simulated Annealing (SA) for fine-tuning and exploration
    
    # Phase 1: Hungarian-based local search (larger moves)
    # We run this for a portion of the time to get into a good basin
    hungarian_deadline = start_time + 30.0  # Spend first 30s on Hungarian
    
    # Precompute A_sym if B is symmetric for faster C construction
    if is_sym:
        A_sym = a + a.T
    else:
        A_sym = None
    
    while time.time() < hungarian_deadline:
        # Construct linear cost matrix C
        # C[i, k] = sum_j a[i, j] b[k, p[j]] + sum_j a[j, i] b[p[j], k]
        # If B is symmetric: C = (A + A.T) @ B_p where B_p[i, :] = B[p[i], :]
        # If B is not symmetric: C = A @ B_p.T + A.T @ B_p
        
        B_p = b[p]  # Rows of B permuted by p. B_p[i, k] = B[p[i], k]
        
        if is_sym:
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p
        
        # Solve linear assignment
        row_ind, col_ind = linear_sum_assignment(C)
        
        # New permutation
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Evaluate new cost
        new_cost = calc_cost(p_new)
        
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement from Hungarian.
            # Try a few random swaps to perturb before next Hungarian iteration
            improved = False
            for _ in range(20):
                i, j = np.random.choice(n, 2, replace=False)
                p_swap = p.copy()
                p_swap[i], p_swap[j] = p_swap[j], p_swap[i]
                swap_cost = calc_cost(p_swap)
                if swap_cost < current_cost:
                    p = p_swap
                    current_cost = swap_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if not improved:
                # Stuck in local optimum for Hungarian. Break to SA phase.
                break

    # Phase 2: Simulated Annealing (fine-grained search)
    # Start SA from the best solution found so far
    p = best_p.copy()
    current_cost = best_cost
    
    # SA Parameters
    T = 500.0
    T_end = 1.0
    # Adjust alpha based on remaining time to ensure we cool down properly
    # Rough estimate: we have ~80s left.
    # Let's do about 500,000 steps.
    # alpha = (T_end / T) ** (1 / n_steps)
    # But we don't know exact steps. Let's use a fixed slow cooling rate.
    alpha = 0.99995
    
    # Precompute for faster delta calculation?
    # For n=256, full recalculation is ~50us.
    # 80s / 50us = 1.6M steps.
    # This is a lot. Let's try to optimize delta calculation if possible, 
    # but full recalculation is safer and simpler.
    
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        p[i], p[j] = p[j], p[i]
        new_cost = calc_cost(p)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            p[i], p[j] = p[j], p[i]
            
        T *= alpha
        if T < T_end:
            T = T_end

    return best_p.tolist()
