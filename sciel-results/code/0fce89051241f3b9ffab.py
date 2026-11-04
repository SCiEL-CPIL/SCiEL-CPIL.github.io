"""QAP solver for tho150 with hybrid approach: linearized assignment + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin before 120s kill

    # Helper: Calculate exact cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    # 1. Initialization: Rank matching
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_sum, kind="stable")
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = central_order
    
    best_cost = calc_cost(p_init)
    best_p = p_init.copy()

    # 2. Linearized Assignment Phase
    # We iterate: compute gradient-based cost matrix, solve Hungarian, local search
    # This allows escaping local minima more effectively than pure local search.

    def get_linearized_cost_matrix(p):
        """Compute a linearized cost matrix for assignment based on current permutation."""
        # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        # Linearize: for each facility i, the cost of assigning to location k is
        # sum_j a[i,j] b[k, p[j]] (approximation, treating other assignments fixed)
        # More precisely, the gradient of the QAP objective with respect to the
        # assignment of facility i to location k (holding others fixed) is:
        # sum_j a[i,j] b[k, p[j]] + sum_j a[j,i] b[p[j], k]
        # This gives a linear cost matrix for the assignment problem.
        
        b_col_p = b[:, p]  # [l, j] = b[l, p[j]]
        b_row_p = b[p, :]  # [j, l] = b[p[j], l]
        
        # cost[i, k] = sum_j a[i,j] b[k, p[j]] + sum_j a[j,i] b[p[j], k]
        # = (a @ b_col_p.T)[i, k] + (a.T @ b_row_p)[i, k]
        
        cost_mat = a @ b_col_p.T + a.T @ b_row_p
        return cost_mat

    def local_search_2opt(p, max_iters=5000, tabu_size=50):
        """Fast 2-opt local search with tabu list."""
        current_cost = calc_cost(p)
        
        # Precompute for fast delta evaluation
        # dot_row[i, l] = sum_k a[i,k] b[l, p[k]]
        # dot_col[i, l] = sum_k a[k,i] b[p[k], l]
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        dot_row = a @ b_col_p.T
        dot_col = a.T @ b_row_p
        b_diag = np.diag(b)
        
        tabu_list = {}
        current_step = 0
        no_improve = 0
        max_no_improve = 200
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            if no_improve > max_no_improve:
                # Perturb
                for _ in range(5):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = calc_cost(p)
                b_col_p = b[:, p]
                b_row_p = b[p, :]
                dot_row = a @ b_col_p.T
                dot_col = a.T @ b_row_p
                no_improve = 0
                tabu_list.clear()
                continue
            
            best_delta = np.inf
            best_i = -1
            best_j = -1
            
            # Full scan of all pairs
            for i in range(n):
                u = p[i]
                for j in range(i + 1, n):
                    key = (i, j)
                    if key in tabu_list and tabu_list[key] > current_step:
                        continue
                    
                    v = p[j]
                    buu = b_diag[u]
                    bvv = b_diag[v]
                    buv = b[u, v]
                    bvu = b[v, u]
                    
                    # Delta calculation using precomputed dot_row and dot_col
                    delta = 0.0
                    # RowPart_i
                    delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvu - buu) - a[i,j]*(bvv - buv)
                    # RowPart_j
                    delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(buv - bvu) - a[j,j]*(buu - bvv)
                    # ColPart_i
                    delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(buv - buu) - a[j,i]*(bvv - bvu)
                    # ColPart_j
                    delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(buu - buv) - a[j,j]*(bvu - bvv)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i == -1:
                no_improve += 1
                continue
            
            # Apply swap
            u, v = p[best_i], p[best_j]
            p[best_i], p[best_j] = v, u
            current_cost += int(best_delta)
            
            # Update dot_row and dot_col incrementally
            delta_row_vec = a[:, best_i] - a[:, best_j]
            delta_p_vec = b[:, v] - b[:, u]
            dot_row += np.outer(delta_row_vec, delta_p_vec)
            
            delta_col_vec = a[best_i, :] - a[best_j, :]
            delta_p_row_vec = b[v, :] - b[u, :]
            dot_col += np.outer(delta_col_vec, delta_p_row_vec)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            tabu_list[(best_i, best_j)] = current_step + tabu_size
            tabu_list[(best_j, best_i)] = current_step + tabu_size
            current_step += 1
            
            if current_step % 50 == 0:
                keys_to_remove = [k for k, val in tabu_list.items() if val <= current_step]
                for k in keys_to_remove:
                    del tabu_list[k]
        
        return best_p, best_cost

    # Main loop: alternate between linearized assignment and local search
    for round_num in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Local search to polish current best
        best_p, best_cost = local_search_2opt(best_p.copy(), max_iters=3000, tabu_size=40)
        
        if time.time() > deadline - 5.0:
            break
        
        # Linearized assignment: compute cost matrix and solve Hungarian
        cost_mat = get_linearized_cost_matrix(best_p)
        
        # Add a small random perturbation to break ties
        if round_num > 0:
            cost_mat = cost_mat + np.random.randn(n, n) * 0.01
        
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        p_candidate = col_ind.copy()
        
        cost_candidate = calc_cost(p_candidate)
        
        # Local search on the new candidate
        p_polished, cost_polished = local_search_2opt(p_candidate.copy(), max_iters=3000, tabu_size=40)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
        else:
            # Try perturbations of best solution
            for _ in range(3):
                p_perturbed = best_p.copy()
                for _ in range(10):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                p_polished, cost_polished = local_search_2opt(p_perturbed, max_iters=2000, tabu_size=30)
                if cost_polished < best_cost:
                    best_cost = cost_polished
                    best_p = p_polished.copy()
                if time.time() > deadline - 5.0:
                    break

    # Final polish with remaining time
    if time.time() < deadline - 1.0:
        best_p, best_cost = local_search_2opt(best_p.copy(), max_iters=10000, tabu_size=60)

    return best_p.tolist()
# EVOLVE-BLOCK-END
