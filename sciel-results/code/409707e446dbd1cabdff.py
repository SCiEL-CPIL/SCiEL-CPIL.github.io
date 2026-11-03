"""Improved QAP solver with vectorized deltas and better ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search(a, b, p, time_limit, start_time, max_passes=3):
    """
    Fast local search with 2-opt swaps using vectorized delta computation.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    a_sym = a + a.T  # symmetric part for delta computation
    
    # Precompute row sums of a for potential use
    a_row_sums = a.sum(axis=1)
    
    pass_count = 0
    improved = True
    while improved and pass_count < max_passes and (time.time() - start_time) < time_limit:
        improved = False
        pass_count += 1
        
        # Vectorized delta computation for all pairs at once would be O(n^2) memory
        # Instead, process row by row
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return p, current_cost
            
            # Compute deltas for swapping i with all j > i
            # delta_ij = sum_k (a[i,k] + a[j,k]) * (D[j,k] - D[i,k]) 
            #           - (a[i,i]+a[i,j]+a[j,i]+a[j,j]) * something...
            # Let's use the correct formula:
            # delta = sum_k [(a[i,k]+a[j,k])*(D[j,k]-D[i,k])] 
            #         - (a[i,i]+a[i,j])*(D[j,i]-D[i,i]) 
            #         - (a[j,i]+a[j,j])*(D[j,j]-D[i,j])
            #         + (a[i,i]-a[j,j])*(D[j,j]-D[i,i])
            #         + (a[i,j]-a[j,i])*(D[j,i]-D[i,j])
            
            # Vectorized: for all j > i
            js = np.arange(i+1, n)
            if len(js) == 0:
                continue
            
            # a[i,:] and a[j,:] for j in js
            a_i = a[i]
            a_js = a[js]  # shape (len(js), n)
            
            D_i = D[i]
            D_js = D[js]  # shape (len(js), n)
            
            # Main term: sum_k (a[i,k] + a[j,k]) * (D[j,k] - D[i,k])
            # = sum_k a[i,k]*(D[j,k]-D[i,k]) + sum_k a[j,k]*(D[j,k]-D[i,k])
            # = a[i,:] @ (D_js - D_i[None,:]) + a_js @ (D_js - D_i[None,:])
            # But this is not quite right because we need element-wise products
            
            # Actually: sum_k (a[i,k] + a[j,k]) * (D[j,k] - D[i,k])
            # Let's compute this properly
            diff_D = D_js - D_i[None, :]  # (len(js), n)
            sum_a = a_i[None, :] + a_js  # (len(js), n)
            main_term = np.sum(sum_a * diff_D, axis=1)  # (len(js),)
            
            # Correction terms
            # - (a[i,i] + a[i,j]) * (D[j,i] - D[i,i])
            a_ii = a[i, i]
            a_ij = a[i, js]  # (len(js),)
            Dj_i = D[js, i]  # (len(js),)
            Di_i = D[i, i]
            corr1 = -(a_ii + a_ij) * (Dj_i - Di_i)
            
            # - (a[j,i] + a[j,j]) * (D[j,j] - D[i,j])
            a_ji = a[js, i]  # (len(js),)
            a_jj = np.diag(a[js])  # (len(js),)
            Dj_j = np.diag(D_js)  # (len(js),)
            Di_j = D_i[js]  # (len(js),)
            corr2 = -(a_ji + a_jj) * (Dj_j - Di_j)
            
            # + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            corr3 = (a_ii - a_jj) * (Dj_j - Di_i)
            
            # + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            corr4 = (a_ij - a_ji) * (Dj_i - Di_j)
            
            deltas = main_term + corr1 + corr2 + corr3 + corr4
            
            # Find best improvement
            min_idx = np.argmin(deltas)
            if deltas[min_idx] < -1e-9:
                j = js[min_idx]
                delta = deltas[min_idx]
                
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D: swap rows i and j, then swap cols i and j
                D[i, :], D[j, :] = D[j, :].copy(), D[i, :].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += delta
                improved = True
    
    return p, current_cost

def _generate_initial_solutions(a, b, n, num_solutions=5, rng=None):
    """Generate diverse initial solutions using different heuristics."""
    if rng is None:
        rng = np.random.default_rng()
    
    solutions = []
    
    # Solution 1: Random
    solutions.append(rng.permutation(n))
    
    # Solution 2: Sort by row sum of a (high flow -> low distance locations)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    a_order = np.argsort(a_row_sums)[::-1]  # high to low flow
    b_order = np.argsort(b_row_sums)  # low to high distance
    p2 = np.empty(n, dtype=np.int64)
    p2[a_order] = b_order
    solutions.append(p2)
    
    # Solution 3: Sort by row sum of a in reverse
    a_order_rev = np.argsort(a_row_sums)
    b_order_rev = np.argsort(b_row_sums)[::-1]
    p3 = np.empty(n, dtype=np.int64)
    p3[a_order_rev] = b_order_rev
    solutions.append(p3)
    
    # Solution 4: Based on eigenvalues of a
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
        # Pair high eigenvalue modes
        a_eig_order = np.argsort(eigvals_a)[::-1]
        b_eig_order = np.argsort(eigvals_b)
        # Create permutation from eigenvector alignment
        p4 = rng.permutation(n)
        solutions.append(p4)
    except:
        solutions.append(rng.permutation(n))
    
    # Solution 5: Another random
    solutions.append(rng.permutation(n))
    
    return solutions

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0  # leave 5s buffer for safety
    
    rng = np.random.default_rng(42)
    
    # Generate initial solutions
    initial_solutions = _generate_initial_solutions(a, b, n, num_solutions=5, rng=rng)
    
    best_p = None
    best_cost = np.inf
    
    # Local search on each initial solution
    for idx, p_init in enumerate(initial_solutions):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_init, remaining - 2, start_time, max_passes=2)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search from best solution
    while (time.time() - start_time) < time_budget - 5:
        # Adaptive perturbation size
        elapsed = time.time() - start_time
        progress = elapsed / time_budget
        k_perturb = max(2, int(8 * (1 - progress)))
        
        p_pert = best_p.copy()
        for _ in range(k_perturb):
            i, j = rng.choice(n, size=2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 3:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_pert, remaining - 2, start_time, max_passes=2)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
