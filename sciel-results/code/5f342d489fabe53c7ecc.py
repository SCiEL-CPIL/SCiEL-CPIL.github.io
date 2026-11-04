import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)
    
    # Precompute symmetric A for delta calculations
    A_sym = a + a.T

    # --- Cost Calculation Helper ---
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    # --- Phase 0: Initialization ---
    # Use spectral embedding for a strong initial guess, with fallback
    try:
        A_sym_init = a + a.T
        B_sym_init = b + b.T

        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym_init)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym_init)

        # Take eigenvectors with largest absolute eigenvalues
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]

        V_A = eigvecs_A[:, idx_A]  # (n, k_eig)
        V_B = eigvecs_B[:, idx_B]  # (n, k_eig)

        # Build assignment cost: align spectral signatures
        C_spectral = -V_A @ V_B.T  # (n, n)

        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind

    except Exception:
        # Fallback to rank-based initialization
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    # --- Vectorized Local Search ---
    def local_search(p, current_cost, time_limit):
        """
        Optimized 2-opt local search for QAP using vectorized delta calculation.
        """
        n = len(p)
        p = p.copy()
        # D[i, j] = b[p[i], p[j]]
        D = b[np.ix_(p, p)].copy()
        current_cost = int(np.sum(a * D))
        
        while time.time() < time_limit:
            improved = False
            # Iterate i from 0 to n-2
            for i in range(n - 1):
                if time.time() > time_limit:
                    return p, current_cost
                
                js = np.arange(i + 1, n)
                
                # Vectorized delta calculation
                diff_A = A_sym[i] - A_sym[js]
                diff_D = D[js] - D[i]
                
                S = np.sum(diff_A * diff_D, axis=1)
                
                # Corrections for diagonal and cross terms
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

    # --- Phase 1: Initial Polish ---
    p, current_cost = local_search(p, current_cost, start_time + 10.0)
    best_p = p.copy()
    best_cost = current_cost
    
    # --- Phase 2: Simulated Annealing / Iterated Local Search Hybrid ---
    # We maintain a 'current' state which can be worse than 'best' to allow escaping local optima.
    current_p = best_p.copy()
    current_cost = best_cost
    
    # SA Parameters
    T = 500.0  # Initial temperature
    cooling_rate = 0.99
    min_T = 1.0
    consecutive_no_improve = 0
    base_perturb = 3

    while time.time() < deadline - 5.0:
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(20, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = current_p.copy()
        n_pert = len(p_pert)
        for _ in range(perturb_k):
            i, j = np.random.choice(n_pert, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Run local search on the perturbed solution
        elapsed = time.time() - start_time
        remaining = deadline - 5.0 - elapsed
        if remaining < 2.0:
            break
            
        # Give LS a fraction of the remaining time
        ls_time = max(1.0, remaining * 0.8)
        ls_end_time = time.time() + ls_time
        
        p_ls, cost_ls = local_search(p_pert, calc_cost(p_pert), ls_end_time)
        
        # Acceptance criterion
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            current_p = p_ls.copy()
            current_cost = cost_ls
            consecutive_no_improve = 0
            T = max(min_T, T * 0.95) # Cool down faster if improved
        else:
            delta = cost_ls - current_cost
            if T > min_T:
                accept_prob = np.exp(-delta / T)
                if np.random.random() < accept_prob:
                    current_p = p_ls.copy()
                    current_cost = cost_ls
            consecutive_no_improve += 1
            T = max(min_T, T * cooling_rate)
            
            # If we've been stuck for a long time, restart from best
            if consecutive_no_improve > 50:
                current_p = best_p.copy()
                current_cost = best_cost
                consecutive_no_improve = 0
                T = max(T, 100.0) # Reheat

    return best_p.tolist()
