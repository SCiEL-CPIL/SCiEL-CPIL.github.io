"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _build_cost_matrix(a, b, p):
    """Build cost matrix C[i,l] = sum_j a[i,j] * b[l, p[j]].
    This gives the marginal cost of placing facility i at location l."""
    n = len(p)
    # C[i,l] = sum_j a[i,j] * b[l, p[j]]
    # V[l,j] = b[l, p[j]], so C = a @ V.T
    V = b[:, p]  # V[l,j] = b[l, p[j]]
    C = a @ V.T
    return C


def _swap_delta_fast(C, a, b, p, i, j):
    """Compute swap delta using precomputed cost matrix C.
    C[i,l] = sum_j a[i,j] * b[l, p[j]] (marginal cost of facility i at location l)
    
    When we swap p[i] and p[j]:
    - Facility i moves from p[i] to p[j]
    - Facility j moves from p[j] to p[i]
    
    The change in total cost:
    delta = (C[i, p[j]] + C[j, p[i]]) - (C[i, p[i]] + C[j, p[j]])
           + correction for the i-j interaction term
    
    Actually, C already includes the interaction with all other facilities.
    The direct a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] term needs special handling.
    
    Let's think more carefully:
    Total cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    After swapping i and j:
    - For pairs (i,k) where k != j: a[i,k]*b[p[j],p[k]] instead of a[i,k]*b[p[i],p[k]]
    - For pairs (j,k) where k != i: a[j,k]*b[p[i],p[k]] instead of a[j,k]*b[p[j],p[k]]
    - For pair (i,j): a[i,j]*b[p[j],p[i]] instead of a[i,j]*b[p[i],p[j]]
    - For pair (j,i): a[j,i]*b[p[i],p[j]] instead of a[j,i]*b[p[j],p[i]]
    
    C[i,l] = sum_k a[i,k] * b[l, p[k]] (sum over all k)
    
    So C[i, p[j]] = sum_k a[i,k] * b[p[j], p[k]]
    C[i, p[i]] = sum_k a[i,k] * b[p[i], p[k]]
    
    The difference C[i, p[j]] - C[i, p[i]] = sum_k a[i,k] * (b[p[j],p[k]] - b[p[i],p[k]])
    This includes k=j term: a[i,j] * (b[p[j],p[j]] - b[p[i],p[j]])
    
    Similarly for j.
    
    The full delta:
    delta = (C[i, p[j]] - C[i, p[i]]) + (C[j, p[i]] - C[j, p[j]])
            + correction for double-counting the i,j interaction
    
    Let's verify: 
    C[i, p[j]] - C[i, p[i]] = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    C[j, p[i]] - C[j, p[j]] = sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    
    Sum = sum_k != i,j [a[i,k]*(b[p[j],p[k]]-b[p[i],p[k]]) + a[j,k]*(b[p[i],p[k]]-b[p[j],p[k]])]
         + a[i,j]*(b[p[j],p[j]]-b[p[i],p[j]]) + a[i,i]*(b[p[j],p[i]]-b[p[i],p[i]])
         + a[j,i]*(b[p[i],p[i]]-b[p[j],p[i]]) + a[j,j]*(b[p[i],p[j]]-b[p[j],p[j]])
    
    The actual delta should be:
    sum_k != i,j [a[i,k]*(b[p[j],p[k]]-b[p[i],p[k]]) + a[j,k]*(b[p[i],p[k]]-b[p[j],p[k]])]
    + a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
    
    So the difference between our computed sum and the actual delta:
    Computed has: a[i,j]*(b[p[j],p[j]]-b[p[i],p[j]]) + a[i,i]*(b[p[j],p[i]]-b[p[i],p[i]])
                  + a[j,i]*(b[p[i],p[i]]-b[p[j],p[i]]) + a[j,j]*(b[p[i],p[j]]-b[p[j],p[j]])
    Actual has:   a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
    
    Correction = Actual - Computed
    = a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) - a[i,j]*(b[p[j],p[j]]-b[p[i],p[j]])
      + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]]) - a[j,i]*(b[p[i],p[i]]-b[p[j],p[i]])
      - a[i,i]*(b[p[j],p[i]]-b[p[i],p[i]])
      - a[j,j]*(b[p[i],p[j]]-b[p[j],p[j]])
    
    This is getting complex. Let me just use the original formula but with the cost matrix for speed.
    
    Actually, let me reconsider. The cost matrix approach:
    C[i,l] = sum_k a[i,k] * b[l, p[k]]
    
    If we swap i and j (pi = p[i], pj = p[j]):
    New C'[i, l] = sum_k a[i,k] * b[l, p'[k]] where p'[i]=pj, p'[j]=pi, p'[k]=p[k] else
    C'[i, l] = sum_{k!=i,j} a[i,k]*b[l,p[k]] + a[i,j]*b[l,pi] + a[i,i]*b[l,pj]
    
    This is getting complicated. Let me just use the efficient delta formula directly.
    """
    pi = p[i]
    pj = p[j]
    # Use the cost matrix for the bulk computation
    # C[i, pj] - C[i, pi] + C[j, pi] - C[j, pj] gives most of the delta
    # but we need to correct for the i,j interaction terms
    
    # The cost matrix C[i,l] = sum_k a[i,k] * b[l, p[k]]
    # When we swap, the "others" change for i and j
    
    # Simplest correct approach: use the original formula
    # But optimize it using the fact that we can precompute row/col sums
    
    # delta = sum_{k != i,j} a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
    #        + sum_{k != i,j} a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    #        + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # = sum_{k != i,j} (a[i,k] - a[j,k])*(b[pj,p[k]] - b[pi,p[k]])
    #   + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # Let's compute using cost matrix differences
    # C[i, pj] = sum_k a[i,k]*b[pj, p[k]]
    # C[i, pi] = sum_k a[i,k]*b[pi, p[k]]
    # C[i,pj] - C[i,pi] = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
    
    # This includes k=i and k=j terms:
    # k=i: a[i,i]*(b[pj,pi] - b[pi,pi])
    # k=j: a[i,j]*(b[pj,pj] - b[pi,pj])
    
    # C[j,pi] - C[j,pj] = sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    # k=i: a[j,i]*(b[pi,pi] - b[pj,pi])
    # k=j: a[j,j]*(b[pi,pj] - b[pj,pj])
    
    # Sum of these two = 
    # sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
    # + a[i,i]*(b[pj,pi]-b[pi,pi]) + a[i,j]*(b[pj,pj]-b[pi,pj])
    # + a[j,i]*(b[pi,pi]-b[pj,pi]) + a[j,j]*(b[pi,pj]-b[pj,pj])
    
    # Actual delta:
    # sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
    # + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    
    # So correction = Actual - Computed
    # = a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #   - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[i,j]*(b[pj,pj]-b[pi,pj])
    #   - a[j,i]*(b[pi,pi]-b[pj,pi]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    
    # This is messy. Let me just use the direct formula with numpy optimizations.
    
    # Direct formula:
    # delta = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    #         - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    #         - a[i,j]*(b[pj,pj]-b[pi,pj]) - a[j,i]*(b[pi,pi]-b[pj,pi])
    #         + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    
    # Simplifying the correction terms:
    # = a[i,j]*(b[pj,pi]-b[pi,pj] - b[pj,pj]+b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi] - b[pi,pi]+b[pj,pi])
    #   - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    # = a[i,j]*(b[pj,pi]-b[pj,pj]) + a[j,i]*(b[pi,pj]-b[pi,pi])
    #   - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    
    # Hmm, this is still complex. Let me just use the straightforward approach:
    
    # delta = (C[i,pj] - C[i,pi]) + (C[j,pi] - C[j,pj])
    #         + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    #         - a[i,i]*(b[pj,pi] - b[pi,pi]) - a[j,j]*(b[pi,pj] - b[pj,pj])
    #         - a[i,j]*(b[pj,pj] - b[pi,pj]) - a[j,i]*(b[pi,pi] - b[pj,pi])
    
    # Let me verify with a simple case... Actually, let me just use the original
    # _swap_delta formula but make it faster by precomputing.
    
    # Original:
    # bpj = b[:, pj][p]  # b[p[k], pj] for all k
    # bpi = b[:, pi][p]  # b[p[k], pi] for all k
    # brj = b[pj, :][p]  # b[pj, p[k]] for all k
    # bri = b[pi, :][p]  # b[pi, p[k]] for all k
    # delta = a[:,i]*(bpj-bpi) + a[:,j]*(bpi-bpj) + a[i,:]*(brj-bri) + a[j,:]*(bri-brj)
    # delta[i] = 0, delta[j] = 0
    # direct = a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    
    # The issue is that a[:,i]*(bpj-bpi) includes k=i and k=j terms that shouldn't be there
    # in the same way. Let me just trust the original formula and optimize the computation.
    
    # For now, use the cost matrix for a fast approximation and verify:
    delta_approx = C[i, pj] - C[i, pi] + C[j, pi] - C[j, pj]
    
    # Correction for the diagonal and cross terms
    # The cost matrix includes all k, but we need to handle i and j specially
    # C[i,pj] includes a[i,i]*b[pj,pi] and a[i,j]*b[pj,pj]
    # C[i,pi] includes a[i,i]*b[pi,pi] and a[i,j]*b[pi,pj]
    # C[j,pi] includes a[j,i]*b[pi,pi] and a[j,j]*b[pi,pj]
    # C[j,pj] includes a[j,i]*b[pj,pi] and a[j,j]*b[pj,pj]
    
    # The correct delta for the i,j interaction is:
    # a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # What we got from cost matrix differences for i,j terms:
    # From C[i,pj]-C[i,pi]: a[i,j]*(b[pj,pj]-b[pi,pj]) + a[i,i]*(b[pj,pi]-b[pi,pi])
    # From C[j,pi]-C[j,pj]: a[j,j]*(b[pi,pj]-b[pj,pj]) + a[j,i]*(b[pi,pi]-b[pj,pi])
    
    # Total from cost matrix for i,j related terms:
    # a[i,j]*(b[pj,pj]-b[pi,pj]) + a[i,i]*(b[pj,pi]-b[pi,pi])
    # + a[j,j]*(b[pi,pj]-b[pj,pj]) + a[j,i]*(b[pi,pi]-b[pj,pi])
    
    # We want: a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    
    # Correction = desired - what_we_have
    # = a[i,j]*(b[pj,pi]-b[pi,pj]) - a[i,j]*(b[pj,pj]-b[pi,pj])
    #   + a[j,i]*(b[pi,pj]-b[pj,pi]) - a[j,i]*(b[pi,pi]-b[pj,pi])
    #   - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    # = a[i,j]*(b[pj,pi]-b[pj,pj]) + a[j,i]*(b[pi,pj]-b[pi,pi])
    #   - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    
    correction = (a[i, j] * (b[pj, pi] - b[pj, pj]) +
                  a[j, i] * (b[pi, pj] - b[pi, pi]) -
                  a[i, i] * (b[pj, pi] - b[pi, pi]) -
                  a[j, j] * (b[pi, pj] - b[pj, pj]))
    
    return int(delta_approx + correction)


def _update_cost_matrix(C, a, b, p, i, j):
    """Update cost matrix C after swapping p[i] and p[j]."""
    pi = p[i]
    pj = p[j]
    
    # After swap: p'[i] = pj, p'[j] = pi
    # C'[k, l] = sum_m a[k,m] * b[l, p'[m]]
    # Only changes for m=i and m=j
    # C'[k, l] = C[k, l] - a[k,i]*b[l, pi] - a[k,j]*b[l, pj] + a[k,i]*b[l, pj] + a[k,j]*b[l, pi]
    # = C[k, l] + a[k,i]*(b[l,pj]-b[l,pi]) + a[k,j]*(b[l,pi]-b[l,pj])
    # = C[k, l] + (a[k,i]-a[k,j])*(b[l,pj]-b[l,pi])
    
    diff = (a[:, i] - a[:, j]) * (b[:, pj] - b[:, pi])  # diff[k] = (a[k,i]-a[k,j])*(b[pj]-b[pi]) for each row l... 
    # Wait, b[:,pj] - b[:,pi] is a vector of length n where element l is b[l,pj]-b[l,pi]
    # (a[:,i]-a[:,j]) is a vector of length n where element k is a[k,i]-a[k,j]
    # We want outer product: C[k,l] += (a[k,i]-a[k,j])*(b[l,pj]-b[l,pi])
    
    C += np.outer(a[:, i] - a[:, j], b[:, pj] - b[:, pi])
    
    # Swap in p
    p[i], p[j] = p[j], p[i]


def _local_search_fast(C, a, b, p, current_cost, deadline, max_passes=50):
    """Fast local search using precomputed cost matrix."""
    n = len(p)
    for _ in range(max_passes):
        if time.time() >= deadline - 1:
            break
        improved = False
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            if time.time() >= deadline - 1:
                break
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                d = _swap_delta_fast(C, a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        _update_cost_matrix(C, a, b, p, best_i, best_j)
        current_cost += best_d
        improved = True
    return p, current_cost, C


def _benders_cut(a, b, p, cut_coeff, cut_const, n):
    """Generate a Benders cut from current assignment p."""
    L = a @ b[:, p].T
    np.minimum(cut_coeff, L, out=cut_coeff)
    return L


def _surrogate_assignment(cut_coeff, n, rng):
    """Assign facilities to locations using the surrogate cost matrix."""
    assigned = np.zeros(n, dtype=bool)
    p = np.empty(n, dtype=np.int64)
    order = rng.permutation(n)
    for idx in order:
        row = cut_coeff[idx].copy()
        row[assigned] = np.inf
        best_loc = np.argmin(row)
        p[idx] = best_loc
        assigned[best_loc] = True
    return p


def _lap_initial(a, b, n):
    """Get initial assignment using LAP on linearized cost."""
    # Linearized cost: minimize sum_i sum_j a[i,j] * b[p[i], p[j]]
    # Approximate by: for each facility i, cost of placing at l is sum_j a[i,j] * b[l, mean_location_j]
    # Better: use the rank-matching or a simple LAP on a surrogate
    
    # Compute a surrogate cost matrix
    # C[i,l] = sum_j a[i,j] * b[l, j] (using identity assignment as proxy)
    C = a @ b
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115
    rng = np.random.default_rng(42)
    
    # Initial assignment: try LAP on surrogate
    p0 = _lap_initial(a, b, n)
    best_p = p0.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Also try rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = _qap_cost(a, b, p_rank)
    if cost_rank < best_cost:
        best_cost = cost_rank
        best_p = p_rank.copy()
    
    # Build initial cost matrix
    C = _build_cost_matrix(a, b, best_p)
    
    # Benders cut storage
    cut_coeff = np.full((n, n), np.inf, dtype=np.float64)
    _benders_cut(a, b, best_p, cut_coeff, 0, n)
    
    # Phase 1: Fast local search from initial solution
    p_ls = best_p.copy()
    cost_ls = best_cost
    C_ls = C.copy()
    p_ls, cost_ls, C_ls = _local_search_fast(C_ls, a, b, p_ls, cost_ls, deadline, max_passes=30)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
        C = C_ls.copy()
        _benders_cut(a, b, best_p, cut_coeff, 0, n)
    
    # Phase 2: SA with fast swap evaluation
    T_start = 3000.0
    T_min = 0.5
    cooling = 0.99997
    max_iters_per_run = 12000
    cut_interval = 500
    
    while time.time() < deadline - 15:
        # Choose starting point
        if rng.random() < 0.25:
            p = _surrogate_assignment(cut_coeff, n, rng)
        else:
            p = best_p.copy()
        
        current_cost = _qap_cost(a, b, p)
        C_sa = _build_cost_matrix(a, b, p)
        T = T_start
        iters_since_cut = 0
        
        for it in range(max_iters_per_run):
            if time.time() >= deadline - 15:
                break
            if T < T_min:
                break
            
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            d = _swap_delta_fast(C_sa, a, b, p, i, j)
            if d < 0 or rng.random() < np.exp(-d / T):
                _update_cost_matrix(C_sa, a, b, p, i, j)
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    C = C_sa.copy()
                    iters_since_cut = 0
            
            iters_since_cut += 1
            T *= cooling
            
            if iters_since_cut >= cut_interval:
                _benders_cut(a, b, best_p, cut_coeff, 0, n)
                iters_since_cut = 0
        
        # Local search polish after each SA run
        if time.time() < deadline - 10:
            p_ls = best_p.copy()
            cost_ls = best_cost
            C_ls = C.copy()
            p_ls, cost_ls, C_ls = _local_search_fast(C_ls, a, b, p_ls, cost_ls, deadline, max_passes=20)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
                C = C_ls.copy()
                _benders_cut(a, b, best_p, cut_coeff, 0, n)
    
    # Final polish: multiple passes of steepest descent
    for _ in range(5):
        if time.time() >= deadline - 3:
            break
        p_pol = best_p.copy()
        cost_pol = best_cost
        C_pol = C.copy()
        p_pol, cost_pol, C_pol = _local_search_fast(C_pol, a, b, p_pol, cost_pol, deadline, max_passes=10)
        if cost_pol < best_cost:
            best_cost = cost_pol
            best_p = p_pol.copy()
            C = C_pol.copy()
        else:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
