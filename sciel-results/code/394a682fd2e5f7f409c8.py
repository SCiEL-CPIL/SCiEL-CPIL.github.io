"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

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
            # delta(i,j) = (a_sym[i] - a_sym[j]) @ (D[j] - D[i]) 
            #            - (a_sym[i,i] - a_sym[j,i]) * (D[j,i] - D[i,i])
            #            - (a_sym[i,j] - a_sym[j,j]) * (D[j,j] - D[i,j])
            #            + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            #            + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            
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
    # Get eigenvalues and eigenvectors of a and b
    # We use the top few eigenvectors to create a score-based assignment
    try:
        # Symmetric part for eigendecomposition
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Get top eigenvectors
        # Use numpy's eigh for symmetric matrices
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
        
        # Assign high score_a to low score_b (or vice versa) - try both and pick better
        # Actually, for minimization, we want to match high flow with low distance
        # But the directive says row sums are NOT strong predictors, so let's try
        # matching based on eigenvector structure
        
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
        cost1 = np.sum(a * b[np.ix_(p, p)])
        cost2 = np.sum(a * b[np.ix_(p_rev, p_rev)])
        
        if cost1 <= cost2:
            return p
        else:
            return p_rev
            
    except:
        return rng.permutation(n)

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
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
    
    initial_solutions = [p_eigen, p_random, p_random2]
    
    # Local search on each initial solution with time division
    time_per_ls = (time_budget - 10) / len(initial_solutions)
    
    for p_init in initial_solutions:
        elapsed = time.time() - start_time
        remaining = time_budget - 10 - elapsed
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init, remaining, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search with adaptive perturbation
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 5:
        elapsed = time.time() - start_time
        remaining = time_budget - 5 - elapsed
        if remaining < 10:
            break
        
        # Adaptive perturbation: increase if stuck
        if consecutive_no_improve > 5:
            perturb_k = min(10, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
        
        p_pert = _perturb(best_p, perturb_k)
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_pert, remaining - 5, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
