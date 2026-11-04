"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _vectorized_deltas(a, a_sum, D, i, n, diag_a, diag_D):
    """
    Compute delta cost for swapping position i with all other positions j.
    Pre-computed: diag_a, diag_D for speed.
    """
    ai_dot_Di = np.dot(a_sum[i], D[i])
    ai_dot_D_all = a_sum[i] @ D
    aj_dot_Dj = np.sum(a_sum * D, axis=1)
    aj_dot_Di = a_sum @ D[i]
    
    S_all = ai_dot_D_all - ai_dot_Di - aj_dot_Dj + aj_dot_Di
    
    coeff_i_all = a_sum[i, i] - a_sum[:, i]
    term_i_all = coeff_i_all * (D[:, i] - D[i, i])
    
    coeff_j_all = a_sum[i, :] - diag_a  # diag_a = np.diag(a_sum)
    term_j_all = coeff_j_all * (diag_D - D[i, :])  # diag_D = np.diag(D)
    
    diag_term_all = (a[i, i] - diag_a[::2]) * (diag_D - D[i, i])  # diag_a[::2] doesn't work, need proper diag
    # Actually let me fix: diag_a should be np.diag(a), not diag of a_sum
    # Let me restructure: pass diag_a_raw = np.diag(a)
    
    cross_term_all = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
    
    deltas = S_all - term_i_all - term_j_all + diag_term_all + cross_term_all
    deltas[i] = 0
    return deltas

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Local search with vectorized delta computation.
    Uses first-improvement for speed, alternating with best-improvement.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    a_sum = a + a.T
    diag_a = np.diag(a).copy()
    
    # Precompute diag_D and update incrementally
    diag_D = np.diag(D).copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            # Compute deltas
            ai_dot_Di = np.dot(a_sum[i], D[i])
            ai_dot_D_all = a_sum[i] @ D
            aj_dot_Dj = np.sum(a_sum * D, axis=1)
            aj_dot_Di = a_sum @ D[i]
            
            S_all = ai_dot_D_all - ai_dot_Di - aj_dot_Dj + aj_dot_Di
            
            coeff_i_all = a_sum[i, i] - a_sum[:, i]
            term_i_all = coeff_i_all * (D[:, i] - D[i, i])
            
            coeff_j_all = a_sum[i, :] - np.diag(a_sum)
            term_j_all = coeff_j_all * (diag_D - D[i, :])
            
            diag_term_all = (a[i, i] - diag_a) * (diag_D - D[i, i])
            
            cross_term_all = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
            
            deltas = S_all - term_i_all - term_j_all + diag_term_all + cross_term_all
            deltas[i] = 0
            
            # Use first-improvement for speed
            best_j = np.argmin(deltas)
            if deltas[best_j] < 0:
                j = best_j
                # Perform swap
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                diag_D[i], diag_D[j] = D[j, j], D[i, i]
                current_cost += deltas[j]
                improved = True
    
    return p, current_cost

def _eigen_init(a, b, rng):
    """
    Initialize using eigenvectors of the matrices.
    For QAP, a good heuristic is to match facilities to locations
    based on principal eigenvectors.
    """
    n = len(a)
    # Get principal eigenvectors
    try:
        _, eigvecs_a = np.linalg.eigh(a)
        _, eigvecs_b = np.linalg.eigh(b)
        # Use the first few eigenvectors to create scores
        # Combine top eigenvectors
        va = np.sum(eigvecs_a[:, -3:], axis=1)  # top 3 eigenvectors
        vb = np.sum(eigvecs_b[:, -3:], axis=1)
        
        # Match facilities to locations by rank of eigen-scores
        fac_rank = np.argsort(va)
        loc_rank = np.argsort(vb)
        p = np.empty(n, dtype=int)
        p[fac_rank] = loc_rank
        return p
    except:
        # Fallback to random
        return rng.permutation(n)

def _rank_match_init(a, b):
    """Initialize by matching facility ranks to location ranks based on row sums."""
    n = len(a)
    fac_order = np.argsort(a.sum(axis=1))
    loc_order = np.argsort(b.sum(axis=1))
    p = np.empty(n, dtype=int)
    p[fac_order] = loc_order
    return p

def _perturb(p, k, rng):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 112  # leave 8s buffer for safety
    
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Eigenvector-based initialization
    p_eig = _eigen_init(a, b, rng)
    inits.append(p_eig)
    
    # 2. Rank-matching initialization
    p_rank = _rank_match_init(a, b)
    inits.append(p_rank)
    
    # 3. Random initialization
    inits.append(rng.permutation(n))
    
    # 4. Another random
    inits.append(rng.permutation(n))
    
    # Run local search on each initialization with time allocation
    init_time_each = min(15, (time_budget - 20) / len(inits))
    
    for p_init in inits:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 10:
            break
        remaining_for_ls = min(init_time_each, time_budget - 10 - elapsed)
        if remaining_for_ls < 3:
            break
        p_ls, cost_ls = _local_search_vectorized(a, b, p_init, remaining_for_ls, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Iterated Local Search with adaptive perturbation
    consecutive_no_improve = 0
    while time.time() - start_time < time_budget - 5:
        # Adaptive perturbation: increase k when stuck
        k = 3 + consecutive_no_improve * 2
        k = min(k, 15)
        p_pert = _perturb(best_p, k, rng)
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search_vectorized(a, b, p_pert, remaining - 3, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            # Occasionally restart from a different initial point
            if consecutive_no_improve >= 5:
                p_new_init = rng.permutation(n)
                rem = time_budget - (time.time() - start_time)
                if rem > 10:
                    p_ls2, cost_ls2 = _local_search_vectorized(a, b, p_new_init, rem - 5, start_time)
                    if cost_ls2 < best_cost:
                        best_cost = cost_ls2
                        best_p = p_ls2
                consecutive_no_improve = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
