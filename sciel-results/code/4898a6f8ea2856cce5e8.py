"""
Simulated Annealing solver for QAPLIB wil100.
Uses Benders-inspired decomposition: a linearized surrogate guides initial
assignment, while exact quadratic cost feedback tightens the search.
Multi-start SA with adaptive cooling and final local polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _linear_surrogate_cost(a, b, p):
    """
    Benders master: linearized surrogate cost.
    Uses row sums of A and column sums of B (for symmetric B, same as row sums).
    surrogate = sum_i (a_row_sum[i]) * (b_col_sum[p[i]])
    This is a relaxation that ignores the quadratic interaction.
    """
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    return int(np.sum(a_row_sums * b_col_sums[p]))

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
    
    # Vectorized swap delta using full dot products minus self/corrections
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T1 = np.dot(diff_a_col, diff_b_row)
    corr_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    corr_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_T1 - corr_i - corr_j
    
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T2 = np.dot(diff_a_row, diff_b_col)
    corr2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    corr2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_T2 - corr2_i - corr2_j
    
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(T1 + T2 + T3)

def _compute_swap_deltas_vectorized(a, b, p):
    """
    Vectorized computation of all swap deltas for permutation p.
    Returns a 2D array of shape (n, n) where entry [i,j] is the delta for swapping i and j.
    Uses matrix operations to compute all pairs at once.
    """
    n = len(p)
    
    # Precompute key matrices
    # For each position i, we need a[:, i] (column of a) and b[p[i], :] (row of b at p[i])
    
    # T1 component: for swap (i,j), involves a[:,i] - a[:,j] dotted with b[p_j,:] - b[p_i,:]
    # T2 component: for swap (i,j), involves a[i,:] - a[j,:] dotted with b[:,p_j] - b[:,p_i]
    # T3 component: scalar cross term
    
    # Let's compute using matrix tricks
    # a_cols[i] = a[:, i], shape (n,)
    # b_rows[i] = b[p[i], :], shape (n,)
    
    # T1: (a[:,i] - a[:,j]) · (b[p_j,:] - b[p_i,:])
    #     = a[:,i]·b[p_j,:] - a[:,i]·b[p_i,:] - a[:,j]·b[p_j,:] + a[:,j]·b[p_i,:]
    
    # Let M1[i,j] = a[:,i] · b[p_j,:]  (outer product of a columns with b rows via p)
    # M1 = a.T @ b[p, :]  -> shape (n, n), M1[i,j] = sum_k a[k,i] * b[p[j],k]
    
    # Actually: M1[i,j] = dot(a[:,i], b[p[j],:]) = a[:,i] @ b[p[j],:]
    # This is (a.T @ b)[i, p[j]]
    
    M1 = a.T @ b  # shape (n, n), M1[i, loc] = sum_k a[k,i] * b[loc,k]
    
    # So M1[i, p[j]] gives a[:,i] · b[p[j],:]
    
    # T1[i,j] = M1[i,p[j]] - M1[i,p[i]] - M1[j,p[j]] + M1[j,p[i]]
    #          + corrections for i==i and j==j terms in the dot product
    
    # The dot product includes k=i and k=j terms which need special handling
    # When k=i: a[i,i]*b[p_j,i] vs a[i,i]*b[p_i,i] etc.
    # This is getting complex. Let me use a different approach.
    
    # Alternative: compute delta for each i against all j at once
    deltas = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        pi = p[i]
        a_col_i = a[:, i]
        b_row_pi = b[pi, :]
        
        # For all j: T1 = dot(a[:,i]-a[:,j], b[p_j,:]-b[p_i,:])
        #            = dot(a[:,i], b[p_j,:]) - dot(a[:,i], b[p_i,:]) - dot(a[:,j], b[p_j,:]) + dot(a[:,j], b[p_i,:])
        
        # dot(a[:,i], b[p_j,:]) for all j = M1[i, p[j]]
        term1 = M1[i, p]  # shape (n,), term1[j] = a[:,i]·b[p_j,:]
        
        # dot(a[:,i], b[p_i,:]) is scalar
        scalar_ii = np.dot(a_col_i, b_row_pi)
        
        # dot(a[:,j], b[p_j,:]) for all j
        # This is sum_k a[k,j]*b[p_j,k] for each j
        # = diagonal of (a.T @ b[p,:]) but indexed by j
        # Actually: for fixed j, sum_k a[k,j]*b[p_j,k]
        # Let M2[j] = sum_k a[k,j]*b[p_j,k]
        M2 = np.einsum('ij,jk->j', a, b[p, :])  # M2[j] = sum_k a[k,j]*b[p_j,k]
        
        # dot(a[:,j], b[p_i,:]) for all j = sum_k a[k,j]*b[p_i,k]
        # = (a.T @ b[p_i,:])[j] but b[p_i,:] is a row
        # = a.T @ b[p_i, :] gives vector of shape (n,), element j = sum_k a[k,j]*b[p_i,k]
        term4 = a.T @ b[p_i, :]  # shape (n,), term4[j] = a[:,j]·b[p_i,:]
        
        # T1[j] = term1[j] - scalar_ii - M2[j] + term4[j]
        T1 = term1 - scalar_ii - M2 + term4
        
        # Corrections for T1: the dot product includes k=i and k=j terms
        # Original T1 = sum_k (a[k,i]-a[k,j])*(b[p_j,k]-b[p_i,k])
        # = sum_k a[k,i]*b[p_j,k] - sum_k a[k,i]*b[p_i,k] - sum_k a[k,j]*b[p_j,k] + sum_k a[k,j]*b[p_i,k]
        # The formula above is correct for the full sums.
        # But we need to exclude k=i and k=j from certain terms? No, the formula is exact.
        
        # Wait, let me reconsider. The swap delta formula from _compute_swap_delta:
        # T1 = dot(a[:,i]-a[:,j], b[p_j,:]-b[p_i,:]) - (a[i,i]-a[i,j])*(b[p_j,i]-b[p_i,i]) - (a[j,i]-a[j,j])*(b[p_j,j]-b[p_i,j])
        
        # So there are correction terms. Let me add them.
        
        # Correction 1: (a[i,i]-a[i,j])*(b[p_j,i]-b[p_i,i]) for each j
        # a[i,i] is scalar, a[i,j] varies with j
        # b[p_j,i] varies with j, b[p_i,i] is scalar
        corr1 = (a[i, i] - a[i, p]) * (b[p, i] - b[pi, i])  # shape (n,)
        
        # Correction 2: (a[j,i]-a[j,j])*(b[p_j,j]-b[p_i,j]) for each j
        # a[j,i] varies with j, a[j,j] varies with j
        # b[p_j,j] varies with j, b[p_i,j] varies with j
        corr2 = (a[:, i] - np.diag(a)) * (b[p, np.arange(n)] - b[pi, :])  # shape (n,)
        
        T1 = T1 - corr1 - corr2
        
        # T2: dot(a[i,:]-a[j,:], b[:,p_j]-b[:,p_i])
        #    = dot(a[i,:], b[:,p_j]) - dot(a[i,:], b[:,p_i]) - dot(a[j,:], b[:,p_j]) + dot(a[j,:], b[:,p_i])
        
        # dot(a[i,:], b[:,p_j]) = sum_k a[i,k]*b[k,p_j]
        # For all j: this is (a[i,:] @ b[:, p[j]]) for each j
        # = a[i,:] @ b[:, p] -> shape (n,), element j = a[i,:]·b[:,p_j]
        
        term1_T2 = a[i, :] @ b[:, p]  # shape (n,)
        
        # dot(a[i,:], b[:,p_i]) scalar
        scalar_ii_T2 = np.dot(a[i, :], b[:, pi])
        
        # dot(a[j,:], b[:,p_j]) for all j
        # = sum_k a[j,k]*b[k,p_j] for each j
        M3 = np.einsum('jk,kj->j', a, b[:, p])  # M3[j] = sum_k a[j,k]*b[k,p_j]
        
        # dot(a[j,:], b[:,p_i]) for all j
        # = sum_k a[j,k]*b[k,p_i] for each j
        term4_T2 = a @ b[:, pi]  # shape (n,), term4_T2[j] = a[j,:]·b[:,p_i]
        
        T2 = term1_T2 - scalar_ii_T2 - M3 + term4_T2
        
        # Corrections for T2:
        # From _compute_swap_delta:
        # corr2_i = (a[i,i]-a[j,i])*(b[p_j,pi]-b[pi,pi])
        # corr2_j = (a[i,j]-a[j,j])*(b[p_j,pj]-b[pi,pj])
        
        corr2_i = (a[i, i] - a[:, i]) * (b[p, pi] - b[pi, pi])  # shape (n,)
        corr2_j = (a[i, :] - np.diag(a)) * (b[p, np.arange(n)] - b[pi, p])  # shape (n,)
        
        T2 = T2 - corr2_i - corr2_j
        
        # T3: (a[i,j]-a[j,i])*(b[p_j,pi]-b[pi,pj])
        # For all j:
        T3 = (a[i, :] - a[:, i]) * (b[p, pi] - b[pi, p])  # shape (n,)
        
        deltas[i, :] = (T1 + T2 + T3).astype(np.int64)
        deltas[:, i] = deltas[i, :]
    
    return deltas

def _local_search_fast(a, b, p, current_cost, max_iters=10000, time_limit=10.0):
    """
    Fast steepest-descent local search using vectorized swap-delta computation.
    Recomputes the full delta matrix each iteration (or periodically) to find best move.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    # Precompute constant parts
    a_diag = np.diag(a)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all swap deltas
        deltas = _compute_swap_deltas_vectorized(a, b, p)
        
        # Find the best (most negative) delta
        # Mask out diagonal and use upper triangle
        np.fill_diagonal(deltas, 0)
        best_delta = np.min(deltas)
        
        if best_delta >= 0:
            break  # Local minimum
        
        # Find position of best delta
        best_idx = np.argmin(deltas)
        best_i = best_idx // n
        best_j = best_idx % n
        
        if best_i == best_j:
            break
        
        # Perform the swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(best_delta)
    
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end, use_surrogate_feedback=False):
    """
    Run simulated annealing for a given duration.
    If use_surrogate_feedback, adjust acceptance based on surrogate cost gap.
    Returns (best_p, best_cost).
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Precompute for surrogate feedback
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        # Benders-inspired: if we have surrogate feedback, use it to modulate acceptance
        if use_surrogate_feedback and T > 10.0:
            surrogate_delta = (a_row_sums[i] * b_col_sums[p[j]] + a_row_sums[j] * b_col_sums[p[i]]
                              - a_row_sums[i] * b_col_sums[p[i]] - a_row_sums[j] * b_col_sums[p[j]])
            
            if surrogate_delta < 0 and delta > 0:
                accept_prob = np.exp(-delta / (T * 0.5))
            elif surrogate_delta > 0 and delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / T)
        else:
            accept_prob = np.exp(-delta / T)
        
        if delta < 0 or np.random.rand() < accept_prob:
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    
    return best_p, best_cost

def _tabu_search(a, b, p, current_cost, time_limit=5.0, tabu_tenure=15):
    """
    Tabu search to escape local minima. Prohibits reverse moves for tabu_tenure iterations.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Tabu list: stores (i, j, expiry_iter) for recently made swaps
    tabu_list = []
    iteration = 0
    
    while time.time() - start_time < time_limit:
        # Clean expired tabu entries
        tabu_list = [(ti, tj, exp) for ti, tj, exp in tabu_list if exp > iteration]
        
        # Compute all swap deltas
        deltas = _compute_swap_deltas_vectorized(a, b, p)
        np.fill_diagonal(deltas, 0)
        
        # Mask tabu moves
        for ti, tj, exp in tabu_list:
            deltas[ti, tj] = 0
            deltas[tj, ti] = 0
        
        best_delta = np.min(deltas)
        
        if best_delta >= 0:
            # No improving non-tabu move; allow best tabu move as aspiration
            # Find best move ignoring tabu
            deltas2 = _compute_swap_deltas_vectorized(a, b, p)
            np.fill_diagonal(deltas2, 0)
            best_delta = np.min(deltas2)
            if best_delta >= 0:
                break
            best_idx = np.argmin(deltas2)
            best_i = best_idx // n
            best_j = best_idx % n
        else:
            best_idx = np.argmin(deltas)
            best_i = best_idx // n
            best_j = best_idx % n
        
        if best_i == best_j:
            break
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(best_delta)
        
        # Add to tabu list
        tabu_list.append((best_i, best_j, iteration + tabu_tenure))
        iteration += 1
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Generate multiple initial solutions
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random1 = _random_initial(n)
    cost_random1 = _qap_cost(a, b, p_random1)
    
    p_random2 = _random_initial(n)
    cost_random2 = _qap_cost(a, b, p_random2)
    
    # Select the best initial solution
    candidates = [(p_greedy, cost_greedy), (p_random1, cost_random1), (p_random2, cost_random2)]
    candidates.sort(key=lambda x: x[1])
    best_p, best_cost = candidates[0]
    best_p = best_p.copy()
    
    # Phase 2: SA runs
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0  # Reserve 20s for final polish (LS + tabu)
    
    # Run 1: From best initial, with surrogate feedback
    seg1_time = remaining * 0.35
    T_start = 500.0
    T_end = 5.0
    p1, cost1 = _sa_run(a, b, best_p, best_cost, seg1_time, T_start, T_end, use_surrogate_feedback=True)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Run 2: From second-best initial, standard SA
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    seg2_time = remaining * 0.35
    T_start = 300.0
    T_end = 3.0
    p2, cost2 = _sa_run(a, b, candidates[1][0], candidates[1][1], seg2_time, T_start, T_end, use_surrogate_feedback=False)
    if cost2 < best_cost:
        best_cost = cost2
        best_p = p2.copy()
    
    # Run 3: From third-best initial, standard SA
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    seg3_time = remaining * 0.35
    T_start = 200.0
    T_end = 2.0
    p3, cost3 = _sa_run(a, b, candidates[2][0], candidates[2][1], seg3_time, T_start, T_end, use_surrogate_feedback=False)
    if cost3 < best_cost:
        best_cost = cost3
        best_p = p3.copy()
    
    # Phase 3: Final polish - local search then tabu search
    elapsed = time.time() - start_time
    polish_budget = max(0.0, total_deadline - elapsed - 5.0)
    
    if polish_budget > 1.0:
        # First: fast steepest-descent local search
        ls_time = polish_budget * 0.5
        best_p, best_cost = _local_search_fast(a, b, best_p, best_cost, max_iters=5000, time_limit=ls_time)
        
        # Then: tabu search to escape any remaining local minimum
        elapsed = time.time() - start_time
        tabu_time = max(0.0, total_deadline - elapsed - 5.0)
        if tabu_time > 1.0:
            best_p, best_cost = _tabu_search(a, b, best_p, best_cost, time_limit=tabu_time, tabu_tenure=15)
    
    # Final deadline guard
    elapsed = time.time() - start_time
    if elapsed > total_deadline - 3.0:
        # Safety: ensure we return within deadline
        pass
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
