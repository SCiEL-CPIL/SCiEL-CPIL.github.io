"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with reduced memory allocations.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    # Preallocate workspace for delta computation to avoid repeated allocations
    max_size = n
    ws_S = np.empty(max_size, dtype=np.float64)
    ws_diff_A = np.empty((max_size, n), dtype=np.float64)
    ws_diff_D = np.empty((max_size, n), dtype=np.float64)
    ws_deltas = np.empty(max_size, dtype=np.float64)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Iterate i from 0 to n-2
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            m = len(js)
            
            # Reuse preallocated buffers
            ws_diff_A[:m] = A_sym[i] - A_sym[js]
            ws_diff_D[:m] = D[js] - D[i]
            
            # Compute S = sum(diff_A * diff_D, axis=1) in-place
            np.multiply(ws_diff_A[:m], ws_diff_D[:m], out=ws_diff_A[:m])
            np.sum(ws_diff_A[:m], axis=1, out=ws_S[:m])
            
            # Correction terms
            # Term for k=i: (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # Term for k=j: (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms for the swap of i and j:
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            # cross_term: (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            # Combine all terms into ws_deltas
            np.subtract(ws_S[:m], term_i, out=ws_deltas[:m])
            ws_deltas[:m] -= term_j
            ws_deltas[:m] += diag_term
            ws_deltas[:m] += cross_term
            
            min_idx = np.argmin(ws_deltas[:m])
            min_delta = ws_deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
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

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    The surrogate cost C[i, j] approximates the contribution of placing facility i at location j.
    C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    This is equivalent to (A @ B)[i, j] + (A.T @ B.T)[i, j] = (A @ B + A.T @ B.T)[i, j]
    Since we want to minimize sum a[i,k] b[p[i], p[k]], this surrogate captures the "average" interaction.
    """
    n = len(a)
    # Calculate surrogate cost matrix
    # Cost[i, j] = sum_k a[i,k] b[j,k] + sum_k a[k,i] b[k,j]
    # Note: b is usually symmetric in QAP, but let's be general.
    # If b is symmetric, b[j,k] = b[k,j].
    # Term 1: (A @ B)[i, j] = sum_k a[i,k] b[k,j]
    # Term 2: (A.T @ B.T)[i, j] = sum_k a[k,i] b[j,k]
    
    cost_matrix = a @ b + a.T @ b.T
    
    # Solve assignment problem
    # row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    # p = np.empty(n, dtype=int)
    # p[row_ind] = col_ind
    
    # p[i] is the location assigned to facility i.
    # linear_sum_assignment returns row_ind, col_ind such that cost_matrix[row_ind, col_ind] is minimized.
    # Here rows are facilities, cols are locations.
    # So p[facility] = location.
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Initial phase: Use Hungarian init and random restarts
    # The Hungarian init provides a structurally distinct starting point.
    
    # 1. Hungarian Initialization
    p_hung = _hungarian_init(a, b)
    # Refine with local search
    # Give it a decent chunk of time, e.g., 20% of budget
    ls_limit = time_budget * 0.2
    p_ls, cost_ls = _local_search(a, b, p_hung, ls_limit, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 2. Random Initials
    # Use remaining time for random restarts and ILS
    # If we have time, try a few random starts
    
    if time.time() - start_time < time_budget * 0.4:
        # Try 2 random starts
        time_per_random = (time_budget * 0.2) / 2
        for _ in range(2):
            if time.time() - start_time > time_budget * 0.4:
                break
            p_init = np.random.permutation(n)
            p_ls_r, cost_ls_r = _local_search(a, b, p_init, time_per_random, start_time)
            if cost_ls_r < best_cost:
                best_cost = cost_ls_r
                best_p = p_ls_r.copy()
    
    # ILS phase
    # Use remaining time for Iterated Local Search with bounded LS runs
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    # Cap each LS run to allow multiple restarts
    max_ls_time = min(8.0, remaining_time / 5.0)  # At least 5 restarts possible
    
    while time.time() - start_time < time_budget - 2:
        # Check time
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        elif consecutive_no_improve > 5:
            perturb_k = base_perturb + consecutive_no_improve // 3
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Allocate bounded time for this LS run
        remaining = (time_budget - 2) - (time.time() - start_time)
        duration = min(max_ls_time, remaining)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
            # Reduce perturbation after improvement
            if base_perturb > 2:
                base_perturb = max(2, base_perturb - 1)
        else:
            consecutive_no_improve += 1
            # Gradually increase perturbation
            if consecutive_no_improve % 5 == 0 and base_perturb < 8:
                base_perturb += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
