import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def calc_cost_full(p):
        return np.sum(a * b[p][:, p])
    
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
    
    def compute_delta_2swap(p, i, j):
        """
        Compute the exact cost delta for swapping facilities i and j in assignment p.
        delta = cost(new_p) - cost(p)
        """
        pi, pj = p[i], p[j]
        
        # The change in cost comes from pairs involving i or j
        # Pairs not involving i or j are unchanged
        # Pairs (i, j) and (j, i): a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] -> a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
        # Change: a[i,j]*(b[pi,pj]-b[pj,pi]) + a[j,i]*(b[pj,pi]-b[pi,pj])
        #        = (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
        
        # Pairs (i, k) for k != i,j: a[i,k]*b[pi, p[k]] -> a[i,k]*b[pj, p[k]]
        # Change: sum_{k!=i,j} a[i,k]*(b[pj, p[k]] - b[pi, p[k]])
        
        # Pairs (k, i) for k != i,j: a[k,i]*b[p[k], pi] -> a[k,i]*b[p[k], pj]
        # Change: sum_{k!=i,j} a[k,i]*(b[p[k], pj] - b[p[k], pi])
        
        # Pairs (j, k) for k != i,j: a[j,k]*b[pj, p[k]] -> a[j,k]*b[pi, p[k]]
        # Change: sum_{k!=i,j} a[j,k]*(b[pi, p[k]] - b[pj, p[k]])
        
        # Pairs (k, j) for k != i,j: a[k,j]*b[p[k], pj] -> a[k,j]*b[p[k], pi]
        # Change: sum_{k!=i,j} a[k,j]*(b[p[k], pi] - b[p[k], pj])
        
        # Let's compute this efficiently
        # delta = (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
        #       + sum_{k!=i,j} [a[i,k]*(b[pj, p[k]] - b[pi, p[k]]) + a[k,i]*(b[p[k], pj] - b[p[k], pi])
        #                        + a[j,k]*(b[pi, p[k]] - b[pj, p[k]]) + a[k,j]*(b[p[k], pi] - b[p[k], pj])]
        
        # Let d_pj_pi = b[pj, :] - b[pi, :]  (shape n)
        # Let d_pi_pj = b[:, pj] - b[:, pi]  (shape n)
        
        # For pairs (i,k): a[i,k] * (b[pj, p[k]] - b[pi, p[k]]) = a[i,k] * d_pj_pi[p[k]]
        # For pairs (k,i): a[k,i] * (b[p[k], pj] - b[p[k], pi]) = a[k,i] * d_pi_pj[p[k]]
        # For pairs (j,k): a[j,k] * (b[pi, p[k]] - b[pj, p[k]]) = -a[j,k] * d_pj_pi[p[k]]
        # For pairs (k,j): a[k,j] * (b[p[k], pi] - b[p[k], pj]) = -a[k,j] * d_pi_pj[p[k]]
        
        # So delta = (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
        #          + sum_{k!=i,j} [ (a[i,k] - a[j,k]) * d_pj_pi[p[k]] + (a[k,i] - a[k,j]) * d_pi_pj[p[k]] ]
        
        d_pj_pi = b[pj, :] - b[pi, :]  # shape (n,)
        d_pi_pj = b[:, pj] - b[:, pi]  # shape (n,)
        
        # Sum over all k, then subtract k=i and k=j terms
        # sum_{k} (a[i,k] - a[j,k]) * d_pj_pi[p[k]]
        # sum_{k} (a[k,i] - a[k,j]) * d_pi_pj[p[k]]
        
        term1_all = np.sum((a[i, :] - a[j, :]) * d_pj_pi[p])
        term2_all = np.sum((a[:, i] - a[:, j]) * d_pi_pj[p])
        
        # Subtract k=i
        term1_i = (a[i, i] - a[j, i]) * d_pj_pi[p[i]]
        term2_i = (a[i, i] - a[i, j]) * d_pi_pj[p[i]]
        
        # Subtract k=j
        term1_j = (a[i, j] - a[j, j]) * d_pj_pi[p[j]]
        term2_j = (a[j, i] - a[j, j]) * d_pi_pj[p[j]]
        
        delta = (a[i, j] - a[j, i]) * (b[pi, pj] - b[pj, pi])
        delta += term1_all - term1_i - term1_j
        delta += term2_all - term2_i - term2_j
        
        return delta
    
    def local_search_2swap(p, cost, max_iters=100):
        """Perform 2-swap local search with vectorized delta computation."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try random 2-swaps
            for _ in range(50):
                if time.time() >= deadline:
                    break
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                
                delta = compute_delta_2swap(p, i, j)
                if delta < 0:
                    old_i, old_j = p[i], p[j]
                    p[i], p[j] = old_j, old_i
                    cost += delta
                    improved = True
        
        return p, cost
    
    def rank_initial():
        """Rank-based initial assignment."""
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p
    
    def random_initial():
        """Random initial assignment."""
        p = np.random.permutation(n)
        return p
    
    def greedy_initial():
        """Greedy initial assignment: place facilities one by one at best location."""
        p = np.full(n, -1, dtype=np.int64)
        remaining_locs = set(range(n))
        
        # Order facilities by total flow (descending)
        a_sums = a.sum(axis=1)
        order = np.argsort(-a_sums, kind='stable')
        
        for idx in order:
            best_loc = -1
            best_cost = np.inf
            for loc in remaining_locs:
                # Cost of placing idx at loc, given current partial assignment
                c = 0
                for k in range(n):
                    if p[k] >= 0:
                        c += a[idx, k] * b[loc, p[k]]
                        c += a[k, idx] * b[p[k], loc]
                if c < best_cost:
                    best_cost = c
                    best_loc = loc
            p[idx] = best_loc
            remaining_locs.remove(best_loc)
        
        return p
    
    # Multi-start local search
    best_p = None
    best_cost = np.inf
    
    num_starts = 5
    
    for start_idx in range(num_starts):
        if time.time() >= deadline:
            break
        
        # Generate initial assignment
        if start_idx == 0:
            p = rank_initial()
        elif start_idx == 1:
            p = greedy_initial()
        else:
            p = random_initial()
        
        current_cost = calc_cost_full(p)
        
        # Local search
        p, current_cost = local_search_2swap(p, current_cost, max_iters=100)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Now use Hungarian-based reassignment to escape local optima
    if best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
    else:
        p = rank_initial()
        current_cost = calc_cost_full(p)
        best_p = p.copy()
        best_cost = current_cost
    
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
            p, current_cost = local_search_2swap(p, current_cost, max_iters=50)
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
                swap_delta = compute_delta_2swap(p, i, j)
                if swap_delta < 0 or (T > min_T and np.random.rand() < np.exp(-swap_delta / T)):
                    old_i, old_j = p[i], p[j]
                    p[i], p[j] = old_j, old_i
                    current_cost += swap_delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    T = max(min_T, T * 0.9)
                else:
                    T = max(min_T, T * 0.98)
    
    return best_p.tolist()
