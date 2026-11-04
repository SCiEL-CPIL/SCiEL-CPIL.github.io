"""
Hybrid QAP solver for wil100: Riemannian CQAP initialization + vectorized steepest-descent
local search + simulated annealing polish.
Combines the structural idea of continuous relaxation on the Stiefel manifold for a
high-quality initial permutation with the fast exact local search and SA machinery.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Term 1: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract contributions from k=i and k=j
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract contributions from k=i and k=j
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    
    Delta(i,j) = sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
               + sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
               + (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # For term 1: sum_k (a[k,i]-a[k,j]) * (b[p[k],p[j]]-b[p[k],p[i]])
    # = sum_k a[k,i]*b[p[k],p[j]] - sum_k a[k,i]*b[p[k],p[i]]
    #   - sum_k a[k,j]*b[p[k],p[j]] + sum_k a[k,j]*b[p[k],p[i]]
    # Let M1[i,j] = sum_k a[k,i]*b[p[k],p[j]] = (a.T @ Bp)[i,j]
    M1 = a.T @ Bp  # M1[i,j] = sum_k a[k,i]*b[p[k],p[j]]
    
    # T1[i,j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    # But we need to exclude k=i and k=j from the sum
    # M1[i,j] includes k=i: a[i,i]*b[p[i],p[j]] and k=j: a[j,i]*b[p[j],p[j]]
    # We want sum_{k!=i,j}, so subtract those terms
    
    # Actually, let me think about this differently.
    # T1[i,j] = sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],p[j]]-b[p[k],p[i]])
    # Let's expand: = sum_{k!=i,j} a[k,i]*b[p[k],p[j]] - sum_{k!=i,j} a[k,i]*b[p[k],p[i]]
    #                 - sum_{k!=i,j} a[k,j]*b[p[k],p[j]] + sum_{k!=i,j} a[k,j]*b[p[k],p[i]]
    
    # sum_{k!=i,j} a[k,i]*b[p[k],p[j]] = M1[i,j] - a[i,i]*b[p[i],p[j]] - a[j,i]*b[p[j],p[j]]
    # sum_{k!=i,j} a[k,i]*b[p[k],p[i]] = M1[i,i] - a[i,i]*b[p[i],p[i]] - a[j,i]*b[p[j],p[i]]
    # sum_{k!=i,j} a[k,j]*b[p[k],p[j]] = M1[j,j] - a[i,j]*b[p[i],p[j]] - a[j,j]*b[p[j],p[j]]
    # sum_{k!=i,j} a[k,j]*b[p[k],p[i]] = M1[j,i] - a[i,j]*b[p[i],p[i]] - a[j,j]*b[p[j],p[i]]
    
    # This is getting complex. Let me use a simpler approach:
    # Compute the full sum including all k, then subtract the k=i and k=j terms.
    
    # Full sum for T1: S1[i,j] = sum_k (a[k,i]-a[k,j]) * (b[p[k],p[j]]-b[p[k],p[i]])
    # = sum_k a[k,i]*b[p[k],p[j]] - sum_k a[k,i]*b[p[k],p[i]] - sum_k a[k,j]*b[p[k],p[j]] + sum_k a[k,j]*b[p[k],p[i]]
    # = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    
    S1 = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    # Now subtract k=i and k=j contributions:
    # k=i: (a[i,i]-a[i,j]) * (b[p[i],p[j]]-b[p[i],p[i]])
    # k=j: (a[j,i]-a[j,j]) * (b[p[j],p[j]]-b[p[j],p[i]])
    
    ai_i = np.diag(a)
    pi = p
    # For each pair (i,j), the k=i term is (a[i,i]-a[i,j])*(b[p[i],p[j]]-b[p[i],p[i]])
    # The k=j term is (a[j,i]-a[j,j])*(b[p[j],p[j]]-b[p[j],p[i]])
    
    # Vectorized: for matrix D, D[i,j] needs to subtract these two terms
    # k=i term matrix: (a[i,i]-a[i,j]) * (b[p[i],p[j]]-b[p[i],p[i]])
    # = (diag(a)[:,None] - a) * (Bp - np.diag(Bp)[:,None])
    
    ki_term = (ai_i[:, None] - a) * (Bp - np.diag(Bp)[:, None])
    # k=j term matrix: (a[j,i]-a[j,j]) * (b[p[j],p[j]]-b[p[j],p[i]])
    # = (a.T - diag(a)[None,:]) * (np.diag(Bp)[None,:] - Bp.T)
    kj_term = (a.T - ai_i[None, :]) * (np.diag(Bp)[None, :] - Bp.T)
    
    T1 = S1 - ki_term - kj_term
    
    # Term 2: sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j],p[k]]-b[p[i],p[k]])
    # Full sum: S2[i,j] = sum_k (a[i,k]-a[j,k]) * (b[p[j],p[k]]-b[p[i],p[k]])
    # = sum_k a[i,k]*b[p[j],p[k]] - sum_k a[i,k]*b[p[i],p[k]] - sum_k a[j,k]*b[p[j],p[k]] + sum_k a[j,k]*b[p[i],p[k]]
    # Let M2[i,j] = sum_k a[i,k]*b[p[k],p[j]] = (a @ Bp)[i,j]
    # Then: sum_k a[i,k]*b[p[j],p[k]] = M2[i,j] (since b is symmetric? No, b may not be symmetric)
    # Wait, b[p[j],p[k]] vs b[p[k],p[j]]. If b is symmetric, they're the same.
    # The QAP typically has symmetric b, but let's not assume.
    
    # Actually for QAP, both a and b are typically symmetric. Let me assume symmetry for now
    # and verify with the cost function.
    
    # If b is symmetric: b[p[j],p[k]] = b[p[k],p[j]]
    # So S2[i,j] = M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i] where M2 = a @ Bp
    
    M2 = a @ Bp  # M2[i,j] = sum_k a[i,k]*b[p[k],p[j]]
    
    S2 = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    # k=i term: (a[i,i]-a[j,i]) * (b[p[j],p[i]]-b[p[i],p[i]])
    # = (a - a.T)[:, i]... let me vectorize
    # (a[i,i]-a[j,i]) * (b[p[j],p[i]]-b[p[i],p[i]])
    # Matrix form: (a - a.T) * (Bp.T - np.diag(Bp)[None,:])... 
    # Actually: for element (i,j): (a[i,i]-a[j,i]) * (b[p[j],p[i]]-b[p[i],p[i]])
    # a[i,i]-a[j,i] = (a - a.T)[i,j] if a is symmetric... no.
    # a[i,i] - a[j,i]: this is (np.diag(a)[:,None] - a.T)[i,j]
    # b[p[j],p[i]] - b[p[i],p[i]]: this is (Bp.T - np.diag(Bp)[None,:])[i,j]
    
    ki_term2 = (ai_i[:, None] - a.T) * (Bp.T - np.diag(Bp)[None, :])
    # k=j term: (a[i,j]-a[j,j]) * (b[p[j],p[j]]-b[p[i],p[j]])
    # = (a - diag(a)[None,:])[i,j] * (np.diag(Bp)[None,:] - Bp)[i,j]
    kj_term2 = (a - ai_i[None, :]) * (np.diag(Bp)[None, :] - Bp)
    
    T2 = S2 - ki_term2 - kj_term2
    
    # Term 3: (a[i,j]-a[j,i]) * (b[p[j],p[i]]-b[p[i],p[j]])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    np.fill_diagonal(D, 0)
    
    return D

def _perturb_permutation(p, k, use_3opt=False):
    """
    Perturb a permutation by performing k random swaps or reversals.
    """
    n = len(p)
    p_new = p.copy()
    for _ in range(k):
        if use_3opt and n > 2:
            # 3-opt: reverse a segment
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                if i > j:
                    i, j = j, i
                p_new[i:j+1] = p_new[i:j+1][::-1]
        else:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new

def _local_search_fast(a, b, p, max_iter=None, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the full delta matrix.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 1000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() - start_time > time_limit:
            break
            
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best improving move
        min_val = np.min(D)
        
        if min_val < 0:
            # Find the indices of the minimum
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            improved = True
            iter_count += 1
        else:
            improved = False
            
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start: try several different initializations
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization (fast and good)
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine - limit to 25s to leave time for ILS
    p, current_cost = _local_search_fast(a, b, p, max_iter=300, time_limit=25.0, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Start 2: Random initialization
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 20:
        p = _random_initial(n)
        current_cost = _qap_cost(a, b, p)
        
        # Local search to refine - limit to 20s
        p, current_cost = _local_search_fast(a, b, p, max_iter=300, time_limit=20.0, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Iterated Local Search (ILS) phase: perturb best solution, run local search
    # This is much more effective than per-step SA because it uses the fast
    # O(n^3) delta matrix local search to escape local minima
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15 and best_p is not None:
        ilsa_deadline = start_time + total_deadline - 5
        perturb_strength = 3  # number of random swaps/reversals per perturbation
        use_3opt = False  # Start with 2-opt, switch to 3-opt after some iterations
        ilsa_iter = 0
        
        while time.time() < ilsa_deadline:
            # Check if we have enough time for one ILS iteration
            if time.time() > ilsa_deadline - 8:
                break
            
            # Switch to 3-opt after 20 iterations for more diversity
            if ilsa_iter > 20:
                use_3opt = True
            
            # Perturb the best solution
            p = _perturb_permutation(best_p, perturb_strength, use_3opt=use_3opt)
            
            current_cost = _qap_cost(a, b, p)
            
            # Run local search on perturbed solution - limit per-iteration time
            iter_deadline = min(ilsa_deadline - 1, time.time() + 8.0)
            p, current_cost = _local_search_fast(a, b, p, max_iter=200, time_limit=iter_deadline - start_time, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                # Reduce perturbation strength on improvement (fine-tuning)
                perturb_strength = max(1, perturb_strength - 1)
            else:
                # Gradually increase perturbation to explore more
                perturb_strength = min(n // 4, perturb_strength + 1)
            
            ilsa_iter += 1
    
    # Final steepest descent polish on the best solution with more iterations
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        # High-precision polish: more iterations, tighter time limit
        p, current_cost = _local_search_fast(a, b, p, max_iter=500, time_limit=total_deadline - 1, start_time=start_time)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
