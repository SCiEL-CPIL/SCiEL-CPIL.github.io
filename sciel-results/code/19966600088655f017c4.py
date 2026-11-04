"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _compute_rc(a, b, p):
    """Precompute RC[i] = sum_k a[i,k]*b[p[i],p[k]] and CC[i] = sum_k a[k,i]*b[p[k],p[i]]."""
    n = len(a)
    bp = b[np.ix_(p, p)]
    rc = a @ bp       # rc[i] = sum_k a[i,k]*b[p[i],p[k]]
    cc = (a.T @ bp)   # cc[i] = sum_k a[k,i]*b[p[k],p[i]]
    return rc, cc


def _update_rc_cc(a, b, p, rc, cc, i, j):
    """Incrementally update rc and cc after swapping positions i and j in p."""
    n = len(a)
    li = p[i]
    lj = p[j]
    
    # For each facility k, rc[k] = sum_m a[k,m]*b[p[k],p[m]]
    # After swap: p[i]=lj, p[j]=li
    # rc[k] changes only if k=i or k=j OR if p[k] is li or lj (i.e., k is i or j)
    # Actually rc[k] depends on p[k] (row index of b) and all p[m] for m.
    # rc[k] = sum_m a[k,m] * b[p[k], p[m]]
    # The only p values that change are p[i] and p[j].
    # So rc[k] changes for all k because b[p[k], p[i]] and b[p[k], p[j]] change.
    # That's O(n^2) which is too slow. Let me think differently.
    
    # Alternative: rc[k] = sum_m a[k,m]*b[p[k],p[m]]
    # = sum_{m != i, j} a[k,m]*b[p[k],p[m]] + a[k,i]*b[p[k],p[i]] + a[k,j]*b[p[k],p[j]]
    # After swap: p[i]=lj, p[j]=li
    # rc_new[k] = sum_{m != i, j} a[k,m]*b[p[k],p[m]] + a[k,i]*b[p[k],lj] + a[k,j]*b[p[k],li]
    # rc_old[k] = sum_{m != i, j} a[k,m]*b[p[k],p[m]] + a[k,i]*b[p[k],li] + a[k,j]*b[p[k],lj]
    # delta_rc[k] = a[k,i]*(b[p[k],lj]-b[p[k],li]) + a[k,j]*(b[p[k],li]-b[p[k],lj])
    #             = (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    
    # Similarly for cc[k] = sum_m a[m,k]*b[p[m],p[k]]
    # cc_new[k] - cc_old[k] = a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[j,k]*(b[li,p[k]]-b[lj,p[k]])
    #                        = (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    
    # But we also need to handle k=i and k=j specially because p[k] itself changes.
    
    # For k=i: p[i] changes from li to lj
    # rc_new[i] = sum_m a[i,m]*b[lj,p[m]] where p[i]=lj, p[j]=li
    # rc_old[i] = sum_m a[i,m]*b[li,p[m]] where p[i]=li, p[j]=lj
    # This is complex. Let me just recompute rc and cc for i and j directly.
    
    # For all k != i, j:
    # delta_rc[k] = (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    # delta_cc[k] = (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    
    # For k=i: recompute directly
    # For k=j: recompute directly
    
    # Efficient vectorized update for all k:
    pk = p
    b_lj_col = b[:, lj]  # b[:,lj]
    b_li_col = b[:, li]  # b[:,li]
    b_lj_row = b[lj, :]  # b[lj,:]
    b_li_row = b[li, :]  # b[li,:]
    
    # For all k: delta_rc[k] = (a[k,i]-a[k,j]) * (b[p[k],lj] - b[p[k],li])
    diff_col = b_lj_col[pk] - b_li_col[pk]  # shape (n,)
    delta_rc = (a[:, i] - a[:, j]) * diff_col
    
    # For all k: delta_cc[k] = (a[i,k]-a[j,k]) * (b[lj,p[k]] - b[li,p[k]])
    diff_row = b_lj_row[pk] - b_li_row[pk]  # shape (n,)
    delta_cc = (a[i, :] - a[j, :]) * diff_row
    
    # Now fix k=i and k=j because p[k] changes for those
    # For k=i: old p[i]=li, new p[i]=lj
    # rc[i] = sum_m a[i,m]*b[p[i],p[m]]
    # Old: sum_m a[i,m]*b[li,p[m]] with p[i]=li, p[j]=lj
    # New: sum_m a[i,m]*b[lj,p[m]] with p[i]=lj, p[j]=li
    # Let's just recompute rc[i] and rc[j] from scratch
    # rc[i] = sum_m a[i,m]*b[p_new[i], p_new[m]]
    # rc[j] = sum_m a[j,m]*b[p_new[j], p_new[m]]
    
    # First apply the vectorized deltas (which are wrong for i and j)
    rc += delta_rc
    cc += delta_cc
    
    # Now recompute rc[i] and rc[j] properly
    # After swap: p[i]=lj, p[j]=li
    # rc[i] = sum_m a[i,m]*b[lj, p[m]]
    # rc[j] = sum_m a[j,m]*b[li, p[m]]
    
    # p is already updated at this point (caller does p[i],p[j]=p[j],p[i] before calling)
    # So p[i]=lj, p[j]=li now
    
    # rc[i] = a[i,:] @ b[lj, p]
    # rc[j] = a[j,:] @ b[li, p]
    rc[i] = a[i, :] @ b[lj, p]
    rc[j] = a[j, :] @ b[li, p]
    
    # Similarly for cc
    # cc[i] = sum_m a[m,i]*b[p[m], lj]
    # cc[j] = sum_m a[m,j]*b[p[m], li]
    cc[i] = a[:, i] @ b[p, lj]
    cc[j] = a[:, j] @ b[p, li]
    
    return rc, cc


def _delta_swap_fast(a, b, p, i, j):
    """Compute delta for swapping facilities i and j. p is current (pre-swap)."""
    li = p[i]
    lj = p[j]
    
    diff_a_row = a[i, :] - a[j, :]
    diff_a_col = a[:, i] - a[:, j]
    
    b_lj_p = b[lj, p]
    b_li_p = b[li, p]
    b_p_lj = b[p, lj]
    b_p_li = b[p, li]
    
    sum1 = diff_a_row @ (b_lj_p - b_li_p)
    sum2 = diff_a_col @ (b_p_lj - b_p_li)
    
    total = sum1 + sum2
    
    # Remove k=i and k=j from the sum (they were included but need special handling)
    total -= (a[i,i]-a[j,i])*(b[lj,li]-b[li,li]) + (a[i,i]-a[i,j])*(b[li,lj]-b[li,li])
    total -= (a[i,j]-a[j,j])*(b[lj,lj]-b[li,lj]) + (a[j,i]-a[j,j])*(b[lj,lj]-b[lj,li])
    
    # Add correct self and cross terms
    self_delta = a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
    cross_delta = a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
    
    total += self_delta + cross_delta
    
    return int(total)


def _initial_permutation(a, b):
    """Generate initial permutation using rank-matching + random restarts."""
    n = len(a)
    best_p = None
    best_cost = np.inf
    
    for trial in range(5):
        if trial == 0:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        else:
            p = np.random.permutation(n).astype(np.int64)
        
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0
    
    p, best_cost = _initial_permutation(a, b)
    current_cost = best_cost
    best_p = p.copy()
    
    # Simulated Annealing
    T = 500.0
    T_min = 0.1
    alpha = 0.999
    
    while T > T_min and time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        if abs(i - j) < n // 4 and np.random.random() < 0.3:
            j = np.random.randint(n)
            if i == j:
                continue
        
        li = p[i]
        lj = p[j]
        delta = _delta_swap_fast(a, b, p, i, j)
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
