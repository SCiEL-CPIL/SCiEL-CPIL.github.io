"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _delta_swap(a_sym, a, D, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j.
    p[i] <-> p[j].
    """
    # Using the formula derived for symmetric QAP approximation:
    # delta = 2 * (a_sym[i] - a_sym[j]) @ (D[j] - D[i])
    #        - (a_sym[i,i] - a_sym[j,i]) * (D[j,i] - D[i,i])
    #        - (a_sym[i,j] - a_sym[j,j]) * (D[j,j] - D[i,j])
    #        + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
    #        + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
    
    # Vectorized computation for single pair
    diff_a = a_sym[i] - a_sym[j]  # shape (n,)
    diff_D = D[j] - D[i]          # shape (n,)
    
    term1 = 2.0 * np.dot(diff_a, diff_D)
    
    term2 = (a_sym[i, i] - a_sym[j, i]) * (D[j, i] - D[i, i])
    term3 = (a_sym[i, j] - a_sym[j, j]) * (D[j, j] - D[i, j])
    term4 = (a[i, i] - a[j, j]) * (D[j, j] - D[i, i])
    term5 = (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
    
    return term1 - term2 - term3 + term4 + term5


def _local_search_2opt(a, b, p, time_limit, start_time):
    """
    First-improvement 2-opt local search.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    a_sym = a + a.T
    
    # Precompute indices for faster access
    while time.time() - start_time < time_limit:
        improved = False
        # Try all pairs, but break early if time is running low
        for i in range(n):
            if time.time() - start_time > time_limit:
                return p, current_cost
            for j in range(i + 1, n):
                delta = _delta_swap(a_sym, a, D, i, j)
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    current_cost += int(delta)
                    improved = True
                    # Restart search from beginning to catch new improvements
                    break
            if improved:
                break
        if not improved:
            break
    
    return p, current_cost


def _eigen_init(a, b, n, rng):
    """
    Eigenvalue-based initialization.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Get top eigenvectors
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        # Take top k eigenvectors
        k = min(8, n)
        idx_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
        idx_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
        
        # Create score vectors
        score_a = np.zeros(n)
        for idx in idx_a:
            score_a += np.abs(eigvals_a[idx]) * np.abs(eigvecs_a[:, idx])
        
        score_b = np.zeros(n)
        for idx in idx_b:
            score_b += np.abs(eigvals_b[idx]) * np.abs(eigvecs_b[:, idx])
        
        # Sort facilities by score_a descending, locations by score_b ascending
        fac_order = np.argsort(score_a)[::-1]
        loc_order = np.argsort(score_b)
        
        # Create permutation
        p = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[i]
        
        return p
    except:
        return rng.permutation(n)


def _hungarian_reopt(a, b, p):
    """
    Hungarian algorithm re-optimization.
    """
    n = len(p)
    B_p = b[p, :]
    M1 = a @ b[:, p].T
    M2 = a.T @ B_p
    C = M1 + M2
    
    row_ind, col_ind = linear_sum_assignment(C)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p


def _perturb(p, k, rng):
    """Randomly swap k pairs."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # 5s safety margin
    
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(123)
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # 1. Eigen-based
    p_eigen = _eigen_init(a, b, n, rng)
    initial_solutions.append(p_eigen)
    
    # 2. Random
    initial_solutions.append(rng.permutation(n))
    
    # 3. Another random
    initial_solutions.append(rng.permutation(n))
    
    # 4. Row-sum based
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rowsum = np.empty(n, dtype=np.int64)
    p_rowsum[heavy] = central
    initial_solutions.append(p_rowsum)
    
    # Local search on each initial solution
    time_per_init = (time_budget - 20) / len(initial_solutions)
    
    for p_init in initial_solutions:
        elapsed = time.time() - start_time
        remaining = time_budget - 20 - elapsed
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search_2opt(a, b, p_init, remaining, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search
    consecutive_no_improve = 0
    base_perturb = 2
    iteration = 0
    
    while time.time() - start_time < time_budget - 10:
        elapsed = time.time() - start_time
        remaining = time_budget - 10 - elapsed
        if remaining < 5:
            break
        
        iteration += 1
        
        # Adaptive perturbation
        if consecutive_no_improve > 4:
            perturb_k = min(8, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
        
        p_pert = _perturb(best_p, perturb_k, rng)
        p_ls, cost_ls = _local_search_2opt(a, b, p_pert, remaining, start_time)
        
        # Occasionally use Hungarian
        if iteration % 4 == 0 and time.time() - start_time < time_budget - 15:
            h_p = _hungarian_reopt(a, b, p_ls)
            h_p, h_cost = _local_search_2opt(a, b, h_p, remaining - 5, start_time)
            if h_cost < cost_ls:
                p_ls, cost_ls = h_p, h_cost
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    # Final polish
    if time.time() - start_time < time_budget - 3:
        final_p, final_cost = _local_search_2opt(a, b, best_p, time_budget - 3, start_time)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
