"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search(a, b, p, deadline):
    """
    Optimized 2-opt local search for QAP.
    deadline: absolute time (from time.time()) at which to stop.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    
    while time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - np.diag(A_sym[js])
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                
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

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    deadline = start_time + time_budget
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple random restarts
    # Use 30% of time for initial exploration
    initial_deadline = start_time + time_budget * 0.3
    
    while time.time() < initial_deadline:
        p_init = np.random.permutation(n)
        # Give each initial LS a small chunk of time, but ensure we don't overshoot
        ls_deadline = min(initial_deadline, time.time() + 2.0)
        p_ls, cost_ls = _local_search(a, b, p_init, ls_deadline)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: Iterated Local Search
    if time.time() >= deadline - 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() < deadline - 2:
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Allocate remaining time to this LS run
        # Reserve 2 seconds for safety
        ls_deadline = deadline - 2
        
        p_ls, cost_ls = _local_search(a, b, p_pert, ls_deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
