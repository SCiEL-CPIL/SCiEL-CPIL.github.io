import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Vectorized computation of all swap deltas for permutation p.
    Returns a 2D array of shape (n, n) where entry [i,j] is the delta for swapping i and j.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i,j] = sum_k a[k,i] * b[p[k], p[j]] = (a.T @ Bp)[i,j]
    # M2[i,j] = sum_k a[i,k] * b[p[k], p[j]] = (a @ Bp)[i,j]
    M1 = a.T @ Bp  # shape (n,n): M1[i,j] = sum_k a[k,i] * Bp[k,j]
    M2 = a @ Bp    # shape (n,n): M2[i,j] = sum_k a[i,k] * Bp[k,j]
    
    # For swap of i and j, the delta involves:
    # Terms where facility i or j changes location.
    # 
    # Let's derive: Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    # After swap: p'[i]=p[j], p'[j]=p[i], p'[k]=p[k] for k!=i,j
    #
    # Delta = sum_{u,v} a[u,v] * (b[p'[u], p'[v]] - b[p[u], p[v]])
    # Only terms where u or v is i or j change.
    #
    # Change from u=i: sum_v a[i,v] * (b[pj, p'[v]] - b[pi, p[v]])
    #   For v=i: a[i,i]*(b[pj,pj]-b[pi,pi])
    #   For v=j: a[i,j]*(b[pj,pi]-b[pi,pj])
    #   For v=k: a[i,k]*(b[pj,pk]-b[pi,pk])
    #
    # Change from v=i: sum_u a[u,i] * (b[p'[u], pj] - b[p[u], pi])
    #   For u=i: a[i,i]*(b[pj,pj]-b[pi,pi])  -- wait, this double counts
    #
    # Let me use the standard formula. The delta for swapping positions i,j in the permutation
    # (i.e., swapping which facilities are at locations p[i] and p[j]) is:
    #
    # Actually, let me use the approach from the inspiration programs that worked.
    # The delta matrix can be computed as:
    #
    # D[i,j] = (M1[i,j] + M1[j,i] - M1[i,i] - M1[j,j]) 
    #        + (M2[i,j] + M2[j,i] - M2[i,i] - M2[j,j])
    #        - 2 * (a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] - a[i,i]*b[p[i],p[i]] - a[j,j]*b[p[j],p[j]])
    #
    # Hmm, this is getting complex. Let me use the verified formula from the current _swap_delta
    # and vectorize it.
    #
    # From _swap_delta:
    # t1 = dot(a[:,i] - a[:,j], b[p,pj] - b[p,pi]) - corrections
    # t2 = dot(a[i,:] - a[j,:], b[pj,p] - b[pi,p]) - corrections
    # t3 = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    #
    # Vectorizing t1: dot(a[:,i]-a[:,j], b[p,pj]-b[p,pi])
    # = (a[:,i]-a[:,j]) @ (b[p,pj]-b[p,pi])
    # Let col_diff = a[:,i] - a[:,j] (vector of length n)
    # Let row_diff = b[p,pj] - b[p,pi] (vector of length n)
    # t1_raw[i,j] = col_diff @ row_diff
    #
    # For all i,j: 
    # t1_raw[i,j] = (a[:,i] - a[:,j]) @ (b[p,pj] - b[p,pi])
    # = a[:,i] @ b[p,pj] - a[:,i] @ b[p,pi] - a[:,j] @ b[p,pj] + a[:,j] @ b[p,pi]
    # = M1[i,j] - M1[i,p.index... ] 
    #
    # Wait, b[p,pj] is the column pj of b indexed by p. So b[p,pj] = b[p, p[j]].
    # a[:,i] @ b[p,pj] = sum_k a[k,i] * b[p[k], p[j]] = M1[i,j]
    # a[:,i] @ b[p,pi] = sum_k a[k,i] * b[p[k], p[i]] = M1[i,i]
    # a[:,j] @ b[p,pj] = sum_k a[k,j] * b[p[k], p[j]] = M1[j,j]
    # a[:,j] @ b[p,pi] = sum_k a[k,j] * b[p[k], p[i]] = M1[j,i]
    #
    # So t1_raw[i,j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    #
    # Similarly for t2:
    # t2_raw[i,j] = (a[i,:] - a[j,:]) @ (b[pj,p] - b[pi,p])
    # = a[i,:] @ b[pj,p] - a[i,:] @ b[pi,p] - a[j,:] @ b[pj,p] + a[j,:] @ b[pi,p]
    # a[i,:] @ b[pj,p] = sum_k a[i,k] * b[pj, p[k]] = M2[i,j]  (since M2[i,j] = sum_k a[i,k]*b[p[k],p[j]], and b is not symmetric in general... wait)
    #
    # Actually M2[i,j] = sum_k a[i,k] * b[p[k], p[j]]. But b[pj, p[k]] != b[p[k], pj] in general.
    # So I need to be careful.
    #
    # Let me define:
    # Bp = b[np.ix_(p,p)] so Bp[k,l] = b[p[k], p[l]]
    # M2[i,j] = sum_k a[i,k] * Bp[k,j] = (a @ Bp)[i,j]
    #
    # a[i,:] @ b[pj, p] = sum_k a[i,k] * b[pj, p[k]]
    # This is NOT M2[i,j] because b[pj, p[k]] != b[p[k], pj] generally.
    #
    # I need a different matrix. Let Bp_T = b[np.ix_(p, p)] but transposed in the b indexing.
    # Actually, b[pj, p[k]] is just b[row=pj, col=p[k]].
    #
    # Let me define Bp_col[j] = b[p, p[j]] which is column j of Bp... no.
    # b[p, p[j]] means: for each k, b[p[k], p[j]]. That's column j of Bp.
    # b[pj, p] means: for each k, b[pj, p[k]]. That's row pj of b indexed by p.
    #
    # So I need:
    # R[i,j] = a[i,:] @ b[pj, p] = sum_k a[i,k] * b[pj, p[k]]
    # This is (a @ Bp_T)[i,j] where Bp_T[k,j] = b[pj, p[k]] = b[p[j], p[k]]
    # So Bp_T = Bp.T? No, Bp.T[k,j] = Bp[j,k] = b[p[j], p[k]]. Yes! Bp_T = Bp.T.
    #
    # So R[i,j] = (a @ Bp.T)[i,j]
    #
    # Let M3 = a @ Bp.T
    # Then t2_raw[i,j] = M3[i,j] - M3[i,i] - M3[j,j] + M3[j,i]
    #
    # Wait, let me recheck:
    # t2_raw[i,j] = a[i,:] @ b[pj,p] - a[i,:] @ b[pi,p] - a[j,:] @ b[pj,p] + a[j,:] @ b[pi,p]
    # = M3[i,j] - M3[i,i] - M3[j,j] + M3[j,i]
    # where M3[i,j] = sum_k a[i,k] * b[p[j], p[k]] = (a @ Bp.T)[i,j]
    #
    # Now for the correction terms in _swap_delta:
    # t1 -= (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])
    # t1 -= (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
    #
    # t2 -= (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
    # t2 -= (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
    #
    # t3 = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    #
    # These corrections are O(1) per pair, so I can compute them vectorized too.
    #
    # Let me define:
    # A_ii = a[np.ix_(p, p)] -- no, a is in facility indices, not location indices.
    # The corrections use a[i,i], a[i,j], etc. which are in facility space.
    # And b[pi,pj], etc. which are in location space.
    #
    # b[pi,pj] = b[p[i], p[j]] = Bp[i,j]
    # b[pi,pi] = Bp[i,i]
    # b[pj,pj] = Bp[j,j]
    # b[pj,pi] = Bp[j,i]
    #
    # So:
    # corr1_1[i,j] = (a[i,i]-a[i,j])*(Bp[i,j]-Bp[i,i])
    # corr1_2[i,j] = (a[j,i]-a[j,j])*(Bp[j,j]-Bp[j,i])
    # corr2_1[i,j] = (a[i,i]-a[j,i])*(Bp[j,i]-Bp[i,i])
    # corr2_2[i,j] = (a[i,j]-a[j,j])*(Bp[j,j]-Bp[i,j])
    # t3[i,j] = (a[i,j]-a[j,i])*(Bp[j,i]-Bp[i,j])
    #
    # These can be computed vectorized using broadcasting.
    
    Bp = b[np.ix_(p, p)]
    M1 = a.T @ Bp      # M1[i,j] = sum_k a[k,i] * Bp[k,j]
    M3 = a @ Bp.T      # M3[i,j] = sum_k a[i,k] * Bp.T[k,j] = sum_k a[i,k] * Bp[j,k] = sum_k a[i,k] * b[p[j],p[k]]
    
    # t1_raw[i,j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    t1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    # t2_raw[i,j] = M3[i,j] - M3[i,i] - M3[j,j] + M3[j,i]
    t2_raw = M3 - np.diag(M3)[:, None] - np.diag(M3)[None, :] + M3.T
    
    # Corrections
    A_ii = np.diag(a)
    A_ii_col = A_ii[:, None]  # a[i,i] for all i
    A_ii_row = A_ii[None, :]  # a[i,i] for all i
    
    # b[pi,pj] = Bp[i,j], b[pi,pi] = Bp[i,i], b[pj,pj] = Bp[j,j], b[pj,pi] = Bp[j,i]
    Bp_ii = np.diag(Bp)
    Bp_ii_col = Bp_ii[:, None]
    Bp_ii_row = Bp_ii[None, :]
    
    # corr1_1[i,j] = (a[i,i]-a[i,j])*(Bp[i,j]-Bp[i,i])
    corr1_1 = (A_ii_col - a) * (Bp - Bp_ii_row)
    
    # corr1_2[i,j] = (a[j,i]-a[j,j])*(Bp[j,j]-Bp[j,i])
    corr1_2 = (a.T - A_ii_row) * (Bp_ii_col - Bp.T)
    
    # corr2_1[i,j] = (a[i,i]-a[j,i])*(Bp[j,i]-Bp[i,i])
    corr2_1 = (A_ii_col - a.T) * (Bp.T - Bp_ii_row)
    
    # corr2_2[i,j] = (a[i,j]-a[j,j])*(Bp[j,j]-Bp[i,j])
    corr2_2 = (a - A_ii_row) * (Bp_ii_col - Bp)
    
    # t3[i,j] = (a[i,j]-a[j,i])*(Bp[j,i]-Bp[i,j])
    t3 = (a - a.T) * (Bp.T - Bp)
    
    D = t1_raw - corr1_1 - corr1_2 + t2_raw - corr2_1 - corr2_2 + t3
    
    # Set diagonal to 0
    np.fill_diagonal(D, 0)
    
    # D should be symmetric: D[i,j] == D[j,i]
    # Due to numerical issues, symmetrize
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    
    return D

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    O(n) time complexity. Used for SA where we only need one delta.
    """
    pi = p[i]
    pj = p[j]
    
    t1 = np.dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi])
    t1 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    t1 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    t2 = np.dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
    t2 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    t2 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    t3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(t1 + t2 + t3)

def _local_search(a, b, p, current_cost, time_limit=10.0):
    """
    Steepest-descent local search using vectorized delta matrix.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        # Find the best (most negative) delta
        i, j = np.unravel_index(np.argmin(D), D.shape)
        best_delta = D[i, j]
        
        if best_delta < 0:
            p[i], p[j] = p[j], p[i]
            cost += int(best_delta)
        else:
            break
            
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end):
    """
    Run simulated annealing for a given duration.
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Initial Solutions
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    best_p = p_greedy.copy() if cost_greedy < cost_random else p_random.copy()
    best_cost = min(cost_greedy, cost_random)
    
    # Phase 2: Local Search Polish
    elapsed = time.time() - start_time
    ls_time = min(15.0, total_deadline - elapsed - 25.0)
    if ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=ls_time)
        
    # Phase 3: Multi-start SA
    elapsed = time.time() - start_time
    sa_time_total = total_deadline - elapsed - 5.0
    num_sa_runs = 5
    sa_time_each = sa_time_total / num_sa_runs
    
    for k in range(num_sa_runs):
        if time.time() - start_time > total_deadline - 5.0:
            break
            
        if k == 0:
            p_start = best_p.copy()
            c_start = best_cost
        else:
            p_start = np.random.permutation(n).astype(np.int64)
            c_start = _qap_cost(a, b, p_start)
            p_start, c_start = _local_search(a, b, p_start, c_start, time_limit=3.0)
        
        T_start = 500.0
        T_end = 5.0
        p_sa, c_sa = _sa_run(a, b, p_start, c_start, sa_time_each, T_start, T_end)
        
        if c_sa < best_cost:
            best_cost = c_sa
            best_p = p_sa.copy()
            
    # Phase 4: Final Polish
    elapsed = time.time() - start_time
    final_ls_time = max(0.0, total_deadline - elapsed - 1.0)
    if final_ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=final_ls_time)
        
    return best_p.tolist()
