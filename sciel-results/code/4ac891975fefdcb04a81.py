import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial assignment: Rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def calc_cost_full(p):
        return np.sum(a * b[p][:, p])
    
    current_cost = calc_cost_full(p)
    best_p = p.copy()
    best_cost = current_cost
    
    def compute_dynamic_cost(p):
        """
        Compute the dynamic cost matrix where C[i,j] represents the
        marginal cost of placing facility i at location j, given that
        all other facilities remain at their current locations.
        C[i,j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        """
        b_jp = b[:, p]  # b_jp[j, k] = b[j, p[k]]
        b_pj = b[p, :]  # b_pj[k, j] = b[p[k], j]
        
        C_first = a @ b_jp.T  # C_first[i, j] = sum_k a[i,k] * b[j, p[k]]
        C_second = a.T @ b_pj  # C_second[i, j] = sum_k a[k,i] * b[p[k], j]
        
        return C_first + C_second
    
    def local_search_2swap(p, cost, max_iters=50):
        """Perform 2-swap local search until no improvement found."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try random 2-swaps
            for _ in range(20):
                if time.time() >= deadline:
                    break
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                old_i, old_j = p[i], p[j]
                p[i], p[j] = old_j, old_i
                new_cost = calc_cost_full(p)
                if new_cost < cost:
                    cost = new_cost
                    improved = True
                else:
                    p[i], p[j] = old_i, old_j
        return p, cost
    
    # Main loop: alternate between Hungarian reassignment and local swaps
    iteration = 0
    T = 5000.0  # Initial temperature
    min_T = 10.0
    
    while time.time() < deadline:
        # Compute dynamic cost matrix
        C = compute_dynamic_cost(p)
        
        # Run Hungarian algorithm to find optimal reassignment
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        # Acceptance criterion:
        # 1. If new_cost < current_cost, accept.
        # 2. If new_cost >= current_cost, accept with probability exp(-delta / T).
        delta = new_cost - current_cost
        if delta < 0 or (T > min_T and np.random.rand() < np.exp(-delta / T)):
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = new_p.copy()
            
            p = new_p.copy()
            current_cost = new_cost
            
            # After accepting a Hungarian move, do local search refinement
            p, current_cost = local_search_2swap(p, current_cost, max_iters=20)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            # Cool down
            T = max(min_T, T * 0.9)
        else:
            # If Hungarian step was rejected, try a random 2-swap to perturb
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
        
        iteration += 1
    
    return best_p.tolist()
