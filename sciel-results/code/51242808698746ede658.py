"""
Spectral-initialized Iterated Local Search solver for QAPLIB wil100.
Strategy:
1. Spectral initialization: align eigenvectors of flow and distance matrices
   to create a geometrically informed initial permutation.
2. Iterated Local Search with fast vectorized swap-delta evaluation.
3. Final polish with a tighter local search pass.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Create an initial permutation using spectral decomposition.
    Aligns the principal eigenvectors of the flow matrix with those of
    the distance matrix to create a geometrically informed assignment.
    """
    n = len(a)
    
    # Symmetrize for eigenvalue computation
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get eigenvectors (sorted by eigenvalue descending)
    eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
    eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
    
    # Sort by eigenvalue descending
    idx_a = np.argsort(-eigvals_a)
    idx_b = np.argsort(-eigvals_b)
    eigvecs_a = eigvecs_a[:, idx_a]
    eigvecs_b = eigvecs_b[:, idx_b]
    
    # Top eigenvector of A (facilities with high mutual flow)
    v_a_top = eigvecs_a[:, 0]
    # Top eigenvector of B (locations with high mutual distance)
    v_b_top = eigvecs_b[:, 0]
    
    # We want high-flow facilities to be at low-distance locations
    fac_order = np.argsort(-v_a_top, kind="stable")
    loc_order = np.argsort(v_b_top, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Greedy heuristic combined with spectral info
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    spectral_score_fac = -v_a_top
    spectral_score_loc = v_b_top
    
    spectral_score_fac = (spectral_score_fac - spectral_score_fac.mean()) / (spectral_score_fac.std() + 1e-10)
    spectral_score_loc = (spectral_score_loc - spectral_score_loc.mean()) / (spectral_score_loc.std() + 1e-10)
    
    row_sum_score_fac = a_row_sums / (a_row_sums.max() + 1e-10)
    row_sum_score_loc = b_row_sums / (b_row_sums.max() + 1e-10)
    
    combined_fac = 0.5 * spectral_score_fac + 0.5 * row_sum_score_fac
    combined_loc = 0.5 * spectral_score_loc + 0.5 * row_sum_score_loc
    
    fac_order2 = np.argsort(-combined_fac, kind="stable")
    loc_order2 = np.argsort(combined_loc, kind="stable")
    
    p2 = np.empty(n, dtype=np.int64)
    p2[fac_order2] = loc_order2
    
    cost1 = _qap_cost(a, b, p)
    cost2 = _qap_cost(a, b, p2)
    
    if cost1 <= cost2:
        return p
    else:
        return p2

def _compute_swap_delta_fast(a, b, p, i, j):
    """
    Fast swap delta for positions i and j in permutation p.
    Uses the fact that we can express delta as:
    delta = sum_{k} (a[k,i] - a[k,j])*(b[p[k],p[j]] - b[p[k],p[i]])
          + sum_{k} (a[i,k] - a[j,k])*(b[p[j],p[k]] - b[p[i],p[k]])
          + (a[i,j]-a[j,i])*(b[p[j],p[i]] - b[p[i],p[j]])
          - corrections for k=i and k=j double counting
    """
    pi = p[i]
    pj = p[j]
    
    # T1: sum_k (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    # = a[:,i] @ (b[p,pj] - b[p,pi]) 
    #   minus corrections for k=i and k=j
    diff_b_col = b[p, pj] - b[p, pi]
    T1 = np.dot(a[:, i], diff_b_col)
    T1 -= (a[i,i] - a[i,j]) * diff_b_col[i]
    T1 -= (a[j,i] - a[j,j]) * diff_b_col[j]
    
    # T2: sum_k (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_b_row = b[pj, p] - b[pi, p]
    T2 = np.dot(a[i, :], diff_b_row)
    T2 -= (a[i,i] - a[j,i]) * diff_b_row[i]
    T2 -= (a[i,j] - a[j,j]) * diff_b_row[j]
    
    # T3
    T3 = (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _local_search_fast(a, b, p, current_cost, max_iter=10000, time_limit=None, start_time=None):
    """
    First-improvement local search with fast swap delta.
    """
    n = len(p)
    improved = True
    iter_count = 0
    deadline = start_time + time_limit if (time_limit is not None and start_time is not None) else None
    
    while improved:
        improved = False
        
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return p, current_cost, iter_count
            
            for j in range(i + 1, n):
                delta = _compute_swap_delta_fast(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    iter_count += 1
                    break
        
        if not improved:
            break
        
        if max_iter is not None and iter_count >= max_iter:
            break
    
    return p, current_cost, iter_count

def _perturb(p, strength=3):
    """Perturb by random swaps."""
    n = len(p)
    p_perturbed = p.copy()
    num_swaps = max(1, strength + np.random.randint(0, 3))
    
    for _ in range(num_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
    
    return p_perturbed

def _simulated_annealing(a, b, p, current_cost, time_limit, start_time):
    """SA polish."""
    n = len(p)
    T_start = 500.0
    T_end = 0.1
    total_deadline = time_limit
    
    while time.time() - start_time < total_deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_delta_fast(a, b, p, i, j)
        new_cost = current_cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / total_deadline)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
    
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Spectral initialization
    p = _spectral_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: Iterated Local Search
    ils_time_limit = total_deadline - 5.0
    
    while time.time() - start_time < start_time + ils_time_limit:
        remaining_ils = ils_time_limit - (time.time() - start_time)
        if remaining_ils < 0.5:
            break
            
        p, current_cost, _ = _local_search_fast(
            a, b, p, current_cost,
            max_iter=2000,
            time_limit=remaining_ils - 0.1,
            start_time=start_time
        )
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        if time.time() - start_time >= start_time + ils_time_limit - 3:
            break
        
        p_perturbed = _perturb(p, strength=3)
        p_perturbed_cost = _qap_cost(a, b, p_perturbed)
        
        if p_perturbed_cost < best_cost * 1.5:
            p = p_perturbed
            current_cost = p_perturbed_cost
        else:
            p = best_p.copy()
            current_cost = best_cost
            p = _perturb(p, strength=2)
            current_cost = _qap_cost(a, b, p)
    
    # Phase 3: Final polish
    p_final = best_p.copy()
    cost_final = best_cost
    
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2:
        p_final, cost_final, _ = _local_search_fast(
            a, b, p_final, cost_final,
            max_iter=5000,
            time_limit=remaining_time - 1,
            start_time=start_time
        )
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 1:
        p_sa, cost_sa = _simulated_annealing(
            a, b, best_p.copy(), best_cost,
            time_limit=remaining_time - 0.5,
            start_time=start_time
        )
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
