"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search_2opt_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps using vectorized delta computation.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    a_sym = a + a.T  # symmetric part
    
    # Precompute for faster delta: we'll use a vectorized approach per iteration
    # For each i, compute delta for all j > i at once
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Vectorized delta computation for swapping i with all j > i
            js = np.arange(i + 1, n)
            
            # diff_a = a_sym[i] - a_sym[js, :]  shape (n-i-1, n)
            diff_a = a_sym[i] - a_sym[js]  # (n-i-1, n)
            # diff_D = D[js, :] - D[i]  shape (n-i-1, n)
            diff_D = D[js] - D[i]  # (n-i-1, n)
            
            # S = sum(diff_a * diff_D, axis=1)
            S = np.sum(diff_a * diff_D, axis=1)
            
            # term_i = (a_sym[i,i] - a_sym[js, i]) * (D[js, i] - D[i, i])
            coeff_i = a_sym[i, i] - a_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # term_j = (a_sym[i, js] - a_sym[js, js]) * (D[js, js] - D[i, js])
            coeff_j = a_sym[i, js] - np.diag(a_sym[js])
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # diag_term = (a[i,i] - a[js,j]) * (D[js,j] - D[i,i])
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            
            # cross_term = (a[i,js] - a[js,i]) * (D[js,i] - D[i,js])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            # Find best improving swap
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
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

def _eigen_init(a, b, n, rng):
    """
    Generate initial permutation using eigenvalue-based heuristic.
    Uses leading eigenvectors of a and b to create a structured initial assignment.
    """
    try:
        # Symmetric part for eigendecomposition
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Get top eigenvectors
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        # Take top k eigenvectors (largest magnitude eigenvalues)
        k = min(16, n)
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
        
        # Create permutation: facility fac_order[i] goes to location loc_order[i]
        p = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[i]
        
        # Also try the reverse matching
        p_rev = np.zeros(n, dtype=int)
        for i in range(n):
            p_rev[fac_order[i]] = loc_order[n - 1 - i]
        
        # Pick the one with lower initial cost
        cost1 = _compute_cost(a, b, p)
        cost2 = _compute_cost(a, b, p_rev)
        
        if cost1 <= cost2:
            return p
        else:
            return p_rev
            
    except:
        return rng.permutation(n)

def _hungarian_step(a, b, p):
    """
    Perform one ADMM-like step: compute gradient of QAP objective
    and solve the linear assignment problem (projection onto Birkhoff polytope).
    """
    n = len(p)
    # Gradient of sum_{i,j} a[i,j] b[p[i],p[j]] w.r.t X
    # C[i,j] = sum_k a[i,k] b[p[k],j] + sum_k a[k,i] b[p[k],j]
    # This is effectively 2 * (A X B) if A,B symmetric.
    
    # Efficient computation of C
    # B_perm[k, j] = b[p[k], j]
    B_perm = b[p, :]
    
    # C1[i, j] = sum_k a[i, k] B_perm[k, j] = (a @ B_perm)[i, j]
    C1 = a @ B_perm
    
    # C2[i, j] = sum_k a[k, i] B_perm[k, j] = (a.T @ B_perm)[i, j]
    C2 = a.T @ B_perm
    
    C = C1 + C2
    
    row_ind, col_ind = linear_sum_assignment(C)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p

def _perturb(p, k, rng):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for safety margin
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse initial solutions
    # 1. Eigenvalue-based initialization
    p_eigen = _eigen_init(a, b, n, rng)
    
    # 2. Random initialization
    p_random = rng.permutation(n)
    
    # 3. Another random
    p_random2 = rng.permutation(n)
    
    # 4. Row-sum based initialization
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rowsum = np.empty(n, dtype=np.int64)
    p_rowsum[heavy] = central
    p_rowsum = p_rowsum.astype(np.int64)
    
    initial_solutions = [p_eigen, p_random, p_random2, p_rowsum]
    
    # Local search on each initial solution with time division
    time_per_ls = (time_budget - 20) / len(initial_solutions)
    
    for p_init in initial_solutions:
        elapsed = time.time() - start_time
        remaining = time_budget - 20 - elapsed
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init, remaining, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search with ADMM (Hungarian) steps
    consecutive_no_improve = 0
    iteration = 0
    
    while time.time() - start_time < time_budget - 5:
        elapsed = time.time() - start_time
        remaining = time_budget - 5 - elapsed
        if remaining < 10:
            break
        
        iteration += 1
        
        # Strategy: Alternate between ADMM step and Perturbation
        # ADMM step provides global move, Perturbation provides local diversity
        
        if iteration % 2 == 0:
            # ADMM Step: Global re-optimization via Linear Assignment
            p_candidate = _hungarian_step(a, b, best_p)
        else:
            # Perturbation: Local diversification
            perturb_k = 3 + (consecutive_no_improve // 3)
            perturb_k = min(perturb_k, 10)
            p_candidate = _perturb(best_p, perturb_k, rng)
        
        # Polish the candidate with local search
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_candidate, remaining - 5, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    # Final polish: one more local search pass on best solution
    if time.time() - start_time < time_budget - 2:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = _local_search_2opt_fast(a, b, final_p, time_budget - 2, start_time)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
