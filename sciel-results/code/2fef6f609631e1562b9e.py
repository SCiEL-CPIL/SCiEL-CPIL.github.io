"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Vectorized Best-Improvement 2-Opt Local Search.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    A_sym = a + a.T
    A_pos = A_sym[p]
    
    # Precompute diagonal terms for correction
    # C_ij = (A_ii - A_jj)(D_jj - D_ii) + (A_ij - A_ji)(D_ji - D_ij)
    # Note: A_ii here is A_sym[i, i] = 2*a[i, i].
    # A_ij is A_sym[i, j] = a[i, j] + a[j, i].
    
    while time.time() - start_time < time_limit:
        # Compute M = A_pos @ D.T
        # M[i, j] = A_pos[i] dot D[j]
        M = A_pos @ D.T
        
        # Base Delta Matrix
        # Delta(i, j) approx M[i, j] - M[i, i] - M[j, j] + M[j, i]
        diag_M = np.diag(M)
        Base_Delta = M - diag_M[:, None] - diag_M[None, :] + M.T
        
        # Correction Matrix C
        # C[i, j] = (A_sym[i, i] - A_sym[j, j]) * (D[j, j] - D[i, i]) + (A_sym[i, j] - A_sym[j, i]) * (D[j, i] - D[i, j])
        # Note: A_sym is symmetric, so A_sym[i, j] == A_sym[j, i]. The second term is 0?
        # Wait. A_sym[i, j] = a[i, j] + a[j, i].
        # In the delta derivation, the cross term was (a[i, j] - a[j, i]).
        # Let's check the derivation again.
        # The term from the sum S includes k=i and k=j.
        # S = sum_k (A_i - A_j) dot (D_j - D_i)
        # This S is NOT the correct delta.
        # The correct delta is S - (terms for k=i) - (terms for k=j) + (actual terms for i, j).
        
        # Let's use the scalar delta for the final step, but use Base_Delta to find top K candidates?
        # Or just iterate i, and vectorize j.
        
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # To speed up, we can just scan i and vectorize j.
        for i in range(n):
            # Vectorized delta for fixed i
            # Delta(i, j) = (A_i - A_j) dot (D_j - D_i) + C_ij
            # = A_i dot D_j - A_i dot D_i - A_j dot D_j + A_j dot D_i + C_ij
            
            # A_i dot D_j for all j: M[i, :]
            # A_i dot D_i: M[i, i]
            # A_j dot D_j for all j: diag_M
            # A_j dot D_i for all j: M[:, i] (since M[j, i] = A_j dot D_i)
            
            vec_delta = M[i, :] - M[i, i] - diag_M + M[:, i]
            
            # Add corrections C[i, j]
            # C[i, j] = (A_sym[i, i] - A_sym[j, j]) * (D[j, j] - D[i, i]) + (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            # Note: The second term uses original a, not A_sym.
            
            # Term 1: (A_sym[i, i] - A_sym[j, j]) * (D[j, j] - D[i, i])
            # A_sym[i, i] is scalar. D[i, i] is scalar.
            # A_sym[j, j] is vector diag_A_sym.
            # D[j, j] is vector diag_D.
            term1 = (A_sym[i, i] - np.diag(A_pos)) * (np.diag(D) - D[i, i])
            
            # Term 2: (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            # a[i, :] - a[:, i] is vector of (a[i, j] - a[j, i])
            # D[:, i] - D[i, :] is vector of (D[j, i] - D[i, j])
            term2 = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
            
            vec_delta += term1 + term2
            
            # Find best j
            min_val = np.min(vec_delta)
            if min_val < best_delta:
                best_delta = min_val
                best_j = np.argmin(vec_delta)
                best_i = i
        
        if best_i != -1 and best_delta < 0:
            # Perform swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            
            # Update D
            D[i], D[j] = D[j].copy(), D[i].copy()
            D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
            
            # Update A_pos
            A_pos[i], A_pos[j] = A_pos[j].copy(), A_pos[i].copy()
            
            current_cost += best_delta
            improved = True
        else:
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
    time_budget = 115
    
    # Random Init
    p = np.random.permutation(n)
    
    # ILS
    best_p = p.copy()
    best_cost = np.inf
    
    while time.time() - start_time < time_budget:
        p_ls, cost_ls = _local_search_vectorized(a, b, p, time_budget, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Perturb
        if time.time() - start_time > time_budget - 5:
            break
        k = random.randint(3, 10)
        p = _perturb(p_ls, k)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
