"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _compute_all_deltas(a, b, p):
    """
    Vectorized computation of delta for all pairs (i, j).
    delta[i, j] = change in cost if we swap facilities at positions i and j.
    
    Uses the identity:
    C(p) = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    After swapping i and j:
    delta_ij = sum_{k != i,j} [a[i,k](b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i](b[p[k],p[j]] - b[p[k],p[i]])]
              + a[i,j](b[p[j],p[i]] - b[p[i],p[j]])
              + a[j,i](b[p[i],p[j]] - b[p[j],p[i]])
              + a[i,i](b[p[j],p[j]] - b[p[i],p[i]])
              + a[j,j](b[p[i],p[i]] - b[p[j],p[j]])
    
    Vectorized approach:
    Let B = b. Define for each position i:
      row_i = B[p[i], :]  (row of B at location p[i])
      col_i = B[:, p[i]]  (col of B at location p[i])
    
    For swap (i,j):
      delta_ij = sum_k [a[i,k](B[p[j],p[k]] - B[p[i],p[k]]) + a[k,i](B[p[k],p[j]] - B[p[k],p[i]])] for k!=i,j
                 + self terms
    
    A cleaner vectorized form:
    Define:
      S_row[i] = B[p[i], p]  (vector of length n: B[p[i], p[0]], B[p[i], p[1]], ...])
      S_col[i] = B[p, p[i]]  (vector of length n: B[p[0], p[i]], B[p[1], p[i]], ...])
    
    Then for swap (i,j):
      delta_ij = sum_k [a[i,k](S_row[j][k] - S_row[i][k]) + a[k,i](S_col[j][k] - S_col[i][k])] for k != i,j
                 + self_terms
    
    Note: S_row[i][k] = B[p[i], p[k]], S_col[i][k] = B[p[k], p[i]]
    
    So delta_ij = sum_k [a[i,k](B[p[j],p[k]] - B[p[i],p[k]]) + a[k,i](B[p[k],p[j]] - B[p[k],p[i]])] for k!=i,j
                 + self
    
    Let's define:
      R_i = B[p[i], p]  -> shape (n,)
      C_i = B[p, p[i]]  -> shape (n,)
    
    Then:
      delta_ij = sum_k != i,j of [a[i,k]*(R_j[k] - R_i[k]) + a[k,i]*(C_j[k] - C_i[k])]
                 + a[i,j]*(R_j[j] - R_i[j]) + a[j,i]*(C_j[i] - C_i[i])
                 + a[i,i]*(R_j[j] - R_i[i]) + a[j,j]*(R_i[i] - R_j[i])
    
    Wait, let me be more careful. The self terms:
      When k=i: a[i,i]*(B[p[j],p[i]] - B[p[i],p[i]]) = a[i,i]*(C_j[i] - C_i[i])... 
      Actually B[p[j],p[i]] = C_j[i] where C_j = B[p, p[j]], so C_j[i] = B[p[i], p[j]].
      And B[p[i],p[i]] = C_i[i].
      
      When k=j: a[j,j]*(B[p[i],p[j]] - B[p[j],p[j]]) = a[j,j]*(C_i[j] - C_j[j])
    
    And the cross terms:
      a[i,j]*(B[p[j],p[i]] - B[p[i],p[j]]) = a[i,j]*(C_j[i] - C_i[j])
      a[j,i]*(B[p[i],p[j]] - B[p[j],p[i]]) = a[j,i]*(C_i[j] - C_j[i])
    
    So total delta_ij = sum_{k!=i,j} [a[i,k]*(R_j[k]-R_i[k]) + a[k,i]*(C_j[k]-C_i[k])]
                       + a[i,i]*(C_j[i]-C_i[i]) + a[j,j]*(C_i[j]-C_j[j])
                       + a[i,j]*(C_j[i]-C_i[j]) + a[j,i]*(C_i[j]-C_j[i])
    
    We can compute this as:
      delta_ij = sum_k [a[i,k]*(R_j[k]-R_i[k]) + a[k,i]*(C_j[k]-C_i[k])]
                 - [terms where k=i and k=j from the sum]
                 + self_terms
    
    The sum over all k includes k=i and k=j. Let's subtract those:
      k=i term: a[i,i]*(R_j[i]-R_i[i]) + a[i,i]*(C_j[i]-C_i[i])
      k=j term: a[i,j]*(R_j[j]-R_i[j]) + a[j,i]*(C_j[j]-C_i[j])
    
    Wait, a[k,i] when k=i is a[i,i], and a[i,k] when k=i is a[i,i].
    When k=j: a[i,j] and a[j,i].
    
    So:
      full_sum_ij = sum_k [a[i,k]*(R_j[k]-R_i[k]) + a[k,i]*(C_j[k]-C_i[k])]
      
      k=i contribution: a[i,i]*(R_j[i]-R_i[i]) + a[i,i]*(C_j[i]-C_i[i])
      k=j contribution: a[i,j]*(R_j[j]-R_i[j]) + a[j,i]*(C_j[j]-C_i[j])
      
      sum_{k!=i,j} = full_sum_ij - k_i_contrib - k_j_contrib
    
    Then:
      delta_ij = sum_{k!=i,j} + self_terms
               = full_sum_ij - a[i,i]*(R_j[i]-R_i[i]) - a[i,i]*(C_j[i]-C_i[i])
                 - a[i,j]*(R_j[j]-R_i[j]) - a[j,i]*(C_j[j]-C_i[j])
                 + a[i,i]*(C_j[i]-C_i[i]) + a[j,j]*(C_i[j]-C_j[j])
                 + a[i,j]*(C_j[i]-C_i[j]) + a[j,i]*(C_i[j]-C_j[i])
    
    Simplify:
      = full_sum_ij 
        - a[i,i]*(R_j[i]-R_i[i])
        - a[i,j]*(R_j[j]-R_i[j]) 
        - a[j,i]*(C_j[j]-C_i[j])
        + a[j,j]*(C_i[j]-C_j[j])
        + a[i,j]*(C_j[i]-C_i[j]) 
        + a[j,i]*(C_i[j]-C_j[i])
    
    Note R_j[i] = B[p[j], p[i]] = C_i[j] (since C_i = B[p, p[i]], so C_i[j] = B[p[j], p[i]])
    And R_i[j] = B[p[i], p[j]] = C_j[i]
    
    So:
      - a[i,j]*(R_j[j]-R_i[j]) = -a[i,j]*(B[p[j],p[j]] - B[p[i],p[j]]) = -a[i,j]*(C_j[j] - C_i[j])
      - a[j,i]*(C_j[j]-C_i[j]) stays
      + a[i,j]*(C_j[i]-C_i[j]) = +a[i,j]*(R_i[j] - R_j[i])... wait C_j[i] = B[p[i],p[j]] = R_i[j], C_i[j] = B[p[j],p[i]] = R_j[i]
      
    This is getting complex. Let me just use a direct vectorized computation.
    
    Alternative: compute delta for all pairs using matrix operations.
    
    delta_ij = sum_k [a[i,k] * (B[p[j],p[k]] - B[p[i],p[k]])] + sum_k [a[k,i] * (B[p[k],p[j]] - B[p[k],p[i]])]
               - correction_terms
    
    Let M1[i,j] = sum_k a[i,k] * (B[p[j],p[k]] - B[p[i],p[k]])
                = sum_k a[i,k]*B[p[j],p[k]] - sum_k a[i,k]*B[p[i],p[k]]
    
    Let v1[i] = sum_k a[i,k]*B[p[i],p[k]]  (this is the "diagonal" contribution for row i)
    Let v2[i,j] = sum_k a[i,k]*B[p[j],p[k]]
    
    Similarly M2[i,j] = sum_k a[k,i] * (B[p[k],p[j]] - B[p[k],p[i]])
    
    This is O(n^3) which is 10^6 for n=100, feasible per iteration.
    
    Actually, let me use a simpler but still vectorized approach for the local search.
    """
    n = len(p)
    # Build the permuted distance matrix
    Bp = b[np.ix_(p, p)]
    
    # For each pair (i,j), compute delta
    # delta_ij = sum_{k != i,j} a[i,k]*(Bp[j,k] - Bp[i,k]) + a[k,i]*(Bp[k,j] - Bp[k,i])
    #           + self terms
    
    # Vectorized: 
    # term1[i,j] = sum_k a[i,k]*(Bp[j,k] - Bp[i,k]) for all k
    # = sum_k a[i,k]*Bp[j,k] - sum_k a[i,k]*Bp[i,k]
    
    # Let A = a, Bp = b[np.ix_(p,p)]
    # term1[i,j] = (A @ Bp.T)[i,j] - (A * Bp[i,:]).sum(axis=1)[i]  ... no
    # sum_k a[i,k]*Bp[j,k] = (A @ Bp.T)[i, j]  because Bp.T[k,j] = Bp[j,k]
    # sum_k a[i,k]*Bp[i,k] = (A * Bp).sum(axis=1)[i]
    
    term1_matrix = a @ Bp.T  # shape (n,n), term1_matrix[i,j] = sum_k a[i,k]*Bp[j,k]
    term1_diag = np.sum(a * Bp, axis=1)  # shape (n,), term1_diag[i] = sum_k a[i,k]*Bp[i,k]
    term1 = term1_matrix - term1_diag[:, None]  # term1[i,j] = sum_k a[i,k]*(Bp[j,k] - Bp[i,k])
    
    # term2[i,j] = sum_k a[k,i]*(Bp[k,j] - Bp[k,i])
    # = sum_k a[k,i]*Bp[k,j] - sum_k a[k,i]*Bp[k,i]
    # sum_k a[k,i]*Bp[k,j] = (a.T @ Bp)[i, j]
    # sum_k a[k,i]*Bp[k,i] = (a.T * Bp.T).sum(axis=0)[i]... 
    # Actually: sum_k a[k,i]*Bp[k,i] = sum_k a.T[i,k]*Bp[k,i] = (a.T * Bp.T).sum(axis=1)[i]... 
    # Let me think: a.T[i,k] = a[k,i]. Bp[k,i] is element (k,i) of Bp.
    # sum_k a.T[i,k] * Bp[k,i] = sum_k a[k,i] * Bp[k,i]
    # This is the element-wise product of a.T and Bp.T, summed over k (axis 0 of Bp, which is axis 1 of Bp.T)
    # Actually: (a.T * Bp.T).sum(axis=1) gives for each i: sum_k a.T[i,k]*Bp.T[k,i] = sum_k a[k,i]*Bp[i,k]
    # Hmm, Bp.T[k,i] = Bp[i,k]. So (a.T * Bp.T).sum(axis=1)[i] = sum_k a[k,i]*Bp[i,k]
    # But I want sum_k a[k,i]*Bp[k,i].
    # Bp[k,i] vs Bp[i,k]: these are different unless Bp is symmetric.
    
    # Let me redo: sum_k a[k,i]*Bp[k,i]
    # = sum_k a[k,i] * Bp[k,i]
    # Let's define this as diag_term2[i]
    diag_term2 = np.sum(a.T * Bp, axis=0)  # a.T has shape (n,n), a.T[i,k]=a[k,i]. Bp[k,i]... 
    # Actually a.T * Bp: element (i,k) is a.T[i,k]*Bp[i,k] = a[k,i]*Bp[i,k]. Sum over k (axis 1) gives sum_k a[k,i]*Bp[i,k].
    # That's not what I want. I want sum_k a[k,i]*Bp[k,i].
    
    # Let me just compute it directly:
    # For each i, sum_k a[k,i]*Bp[k,i] = sum_k a.T[i,k]*Bp[k,i]
    # Bp[k,i] is column i of Bp.
    # So it's the dot product of a.T[i,:] with Bp[:,i]
    # = (a.T * Bp.T).sum(axis=1)... no.
    # a.T[i,:] = a[:,i]. Bp[:,i] is column i.
    # dot(a[:,i], Bp[:,i]) = sum_k a[k,i]*Bp[k,i]
    # This can be computed as: np.sum(a * Bp.T, axis=0)  -> a[i,k]*Bp.T[k,j] = a[i,k]*Bp[j,k]. Sum over k: sum_k a[i,k]*Bp[j,k]. That's term1_matrix[i,j].
    
    # I'm getting confused. Let me just compute term2 differently.
    # term2[i,j] = sum_k a[k,i]*(Bp[k,j] - Bp[k,i])
    # = sum_k a[k,i]*Bp[k,j] - sum_k a[k,i]*Bp[k,i]
    
    # sum_k a[k,i]*Bp[k,j] = (a.T @ Bp)[i,j]  because a.T[i,k]=a[k,i], so (a.T @ Bp)[i,j] = sum_k a[k,i]*Bp[k,j]. Yes!
    term2_matrix = a.T @ Bp  # term2_matrix[i,j] = sum_k a[k,i]*Bp[k,j]
    
    # sum_k a[k,i]*Bp[k,i]:
    # = sum_k a.T[i,k]*Bp[k,i]
    # Let's compute this as: for each i, np.dot(a[:,i], Bp[:,i])
    # Vectorized: np.sum(a * Bp, axis=0) gives sum_k a[k,i]*Bp[k,i]? 
    # a[k,i]*Bp[k,i]: sum over k. a has shape (n,n), Bp has shape (n,n).
    # np.sum(a * Bp, axis=0)[i] = sum_k a[k,i]*Bp[k,i]. Yes!
    term2_diag = np.sum(a * Bp, axis=0)  # term2_diag[i] = sum_k a[k,i]*Bp[k,i]
    term2 = term2_matrix - term2_diag[None, :]  # term2[i,j] = sum_k a[k,i]*(Bp[k,j] - Bp[k,i])
    
    # Now delta[i,j] for i != j:
    # = term1[i,j] + term2[i,j] - correction for k=i and k=j in the sums
    # 
    # In term1[i,j] = sum_k a[i,k]*(Bp[j,k]-Bp[i,k]), the k=i term is a[i,i]*(Bp[j,i]-Bp[i,i]) and k=j term is a[i,j]*(Bp[j,j]-Bp[i,j])
    # In term2[i,j] = sum_k a[k,i]*(Bp[k,j]-Bp[k,i]), the k=i term is a[i,i]*(Bp[i,j]-Bp[i,i]) and k=j term is a[j,i]*(Bp[j,j]-Bp[j,i])
    #
    # The actual delta should NOT include these k=i and k=j terms in the "sum over k!=i,j" part, but instead have specific self terms.
    #
    # Correct delta_ij = sum_{k!=i,j} [a[i,k]*(Bp[j,k]-Bp[i,k]) + a[k,i]*(Bp[k,j]-Bp[k,i])]
    #                   + a[i,i]*(Bp[j,j]-Bp[i,i]) + a[j,j]*(Bp[i,i]-Bp[j,j])
    #                   + a[i,j]*(Bp[j,i]-Bp[i,j]) + a[j,i]*(Bp[i,j]-Bp[j,i])
    #
    # = [term1[i,j] - a[i,i]*(Bp[j,i]-Bp[i,i]) - a[i,j]*(Bp[j,j]-Bp[i,j])]
    #   + [term2[i,j] - a[i,i]*(Bp[i,j]-Bp[i,i]) - a[j,i]*(Bp[j,j]-Bp[j,i])]
    #   + a[i,i]*(Bp[j,j]-Bp[i,i]) + a[j,j]*(Bp[i,i]-Bp[j,j])
    #   + a[i,j]*(Bp[j,i]-Bp[i,j]) + a[j,i]*(Bp[i,j]-Bp[j,i])
    #
    # Simplify:
    # = term1[i,j] + term2[i,j]
    #   - a[i,i]*Bp[j,i] + a[i,i]*Bp[i,i] - a[i,j]*Bp[j,j] + a[i,j]*Bp[i,j]
    #   - a[i,i]*Bp[i,j] + a[i,i]*Bp[i,i] - a[j,i]*Bp[j,j] + a[j,i]*Bp[j,i]
    #   + a[i,i]*Bp[j,j] - a[i,i]*Bp[i,i] + a[j,j]*Bp[i,i] - a[j,j]*Bp[j,j]
    #   + a[i,j]*Bp[j,i] - a[i,j]*Bp[i,j] + a[j,i]*Bp[i,j] - a[j,i]*Bp[j,i]
    #
    # Combine terms:
    # a[i,i]: +Bp[i,i] + Bp[i,i] - Bp[i,i] = +Bp[i,i]  (from: +a[i,i]*Bp[i,i] + a[i,i]*Bp[i,i] - a[i,i]*Bp[i,i])
    #   Wait let me list all a[i,i] terms:
    #   -a[i,i]*Bp[j,i] + a[i,i]*Bp[i,i] - a[i,i]*Bp[i,j] + a[i,i]*Bp[i,i] + a[i,i]*Bp[j,j] - a[i,i]*Bp[i,i]
    #   = a[i,i]*(Bp[i,i] + Bp[i,i] - Bp[i,i] - Bp[j,i] - Bp[i,j] + Bp[j,j])
    #   = a[i,i]*(Bp[i,i] - Bp[j,i] - Bp[i,j] + Bp[j,j])
    #
    # a[i,j] terms:
    #   -a[i,j]*Bp[j,j] + a[i,j]*Bp[i,j] + a[i,j]*Bp[j,i] - a[i,j]*Bp[i,j]
    #   = a[i,j]*(Bp[j,i] - Bp[j,j])
    #
    # a[j,i] terms:
    #   -a[j,i]*Bp[j,j] + a[j,i]*Bp[j,i] + a[j,i]*Bp[i,j] - a[j,i]*Bp[j,i]
    #   = a[j,i]*(Bp[i,j] - Bp[j,j])
    #
    # a[j,j] terms:
    #   +a[j,j]*Bp[i,i] - a[j,j]*Bp[j,j]
    #   = a[j,j]*(Bp[i,i] - Bp[j,j])
    #
    # So:
    # delta[i,j] = term1[i,j] + term2[i,j]
    #   + a[i,i]*(Bp[i,i] - Bp[j,i] - Bp[i,j] + Bp[j,j])
    #   + a[i,j]*(Bp[j,i] - Bp[j,j])
    #   + a[j,i]*(Bp[i,j] - Bp[j,j])
    #   + a[j,j]*(Bp[i,i] - Bp[j,j])
    
    # For i == j, delta[i,i] = 0 (no swap)
    
    delta = term1 + term2
    
    # Add correction terms
    # a[i,i]*(Bp[i,i] - Bp[j,i] - Bp[i,j] + Bp[j,j])
    diag_a_ii = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # Term: a[i,i]*(Bp[i,i] - Bp[j,i] - Bp[i,j] + Bp[j,j])
    # = a[i,i]*Bp[i,i] - a[i,i]*Bp[j,i] - a[i,i]*Bp[i,j] + a[i,i]*Bp[j,j]
    corr1 = diag_a_ii[:, None] * diag_Bp[:, None]  # a[i,i]*Bp[i,i]
    corr1 -= diag_a_ii[:, None] * Bp.T  # -a[i,i]*Bp[j,i]  (Bp.T[i,j] = Bp[j,i])
    corr1 -= diag_a_ii[:, None] * Bp  # -a[i,i]*Bp[i,j]
    corr1 += diag_a_ii[:, None] * diag_Bp[None, :]  # +a[i,i]*Bp[j,j]
    
    # Term: a[i,j]*(Bp[j,i] - Bp[j,j])
    corr2 = a * (Bp.T - diag_Bp[None, :])
    
    # Term: a[j,i]*(Bp[i,j] - Bp[j,j])
    corr3 = a.T * (Bp - diag_Bp[None, :])
    
    # Term: a[j,j]*(Bp[i,i] - Bp[j,j])
    corr4 = diag_a_ii[None, :] * (diag_Bp[:, None] - diag_Bp[None, :])
    
    delta += corr1 + corr2 + corr3 + corr4
    
    # Set diagonal to 0 (no self-swap)
    np.fill_diagonal(delta, 0.0)
    
    return delta


def _local_search(a, b, p, deadline, max_passes=50):
    """
    Best-improvement local search using vectorized delta computation.
    """
    n = len(p)
    cost = _cost(a, b, p)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        delta = _compute_all_deltas(a, b, p)
        
        # Find the best improving swap
        min_val = delta.min()
        if min_val >= 0:
            break
        
        # Get indices of the best swap
        min_idx = np.unravel_index(np.argmin(delta), delta.shape)
        i, j = min_idx
        if i == j:
            break
        
        p[i], p[j] = p[j], p[i]
        cost += int(min_val)
    
    return p, cost


def _sa_run(a, b, p, deadline, initial_temp=1000.0, cooling=0.95, max_iters=5000):
    """
    Simulated annealing with vectorized delta computation for evaluation.
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_cost = cost
    best_p = p.copy()
    temp = initial_temp
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        # Compute delta for this specific swap
        # Use the vectorized delta computation but only extract what we need
        # For efficiency, compute delta for all and pick (i,j)
        # Actually, for SA we just need one delta, so let's compute it directly
        pi = p[i]
        pj = p[j]
        
        # Direct delta computation for swap (i,j)
        # delta = sum_{k!=i,j} [a[i,k]*(B[pj,pk]-B[pi,pk]) + a[k,i]*(B[pk,pj]-B[pk,pi])]
        #         + a[i,i]*(B[pj,pj]-B[pi,pi]) + a[j,j]*(B[pi,pi]-B[pj,pj])
        #         + a[i,j]*(B[pj,pi]-B[pi,pj]) + a[j,i]*(B[pi,pj]-B[pj,pi])
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        B_mask = b[np.ix_(p[mask], p[mask])]
        diff1 = b[pj, p[mask]] - b[pi, p[mask]]
        diff2 = b[p[mask], pj] - b[p[mask], pi]
        
        a_row_i = a[i, mask]
        a_col_i = a[mask, i]
        
        delta_k = np.sum(a_row_i * diff1) + np.sum(a_col_i * diff2)
        
        delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                      + a[j, i] * (b[pi, pj] - b[pj, pi])
                      + a[i, i] * (b[pj, pj] - b[pi, pi])
                      + a[j, j] * (b[pi, pi] - b[pj, pj]))
        
        delta = int(delta_k + delta_self)
        
        new_cost = cost + delta
        
        if new_cost < cost or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        temp *= cooling
    
    return best_p, best_cost


def _hungarian_init(a, b, n, rng):
    """
    Generate initial permutation using Hungarian algorithm on a surrogate cost matrix.
    """
    # Use row sums of a and b to create a surrogate
    a_row = a.sum(axis=1)
    b_col = b.sum(axis=0)
    
    # Create surrogate cost matrix: high flow should go to low distance
    surrogate = -np.outer(a_row, b_col)
    
    row_ind, col_ind = linear_sum_assignment(surrogate)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def _random_init(n, rng):
    """Random permutation."""
    return rng.permutation(n).astype(np.int64)


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization using eigenvectors.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)
    
    k = min(10, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]
    
    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]
    
    fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
    fac_norm[fac_norm == 0] = 1
    fac_emb = fac_emb / fac_norm
    
    loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
    loc_norm[loc_norm == 0] = 1
    loc_emb = loc_emb / loc_norm
    
    cost_matrix = -fac_emb @ loc_emb.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 150s kill
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Multiple initializations, each refined by local search
    initializations = []
    
    # Init 1: Hungarian on surrogate
    p1 = _hungarian_init(a, b, n, rng)
    initializations.append(p1)
    
    # Init 2: Spectral
    p2 = _spectral_init(a, b, n, rng)
    initializations.append(p2)
    
    # Init 3: Random
    p3 = _random_init(n, rng)
    initializations.append(p3)
    
    # Init 4: Reverse Hungarian
    a_row = a.sum(axis=1)
    b_col = b.sum(axis=0)
    surrogate = np.outer(a_row, b_col)  # Positive, so minimizes high-flow to high-distance
    row_ind, col_ind = linear_sum_assignment(surrogate)
    p4 = np.empty(n, dtype=np.int64)
    p4[row_ind] = col_ind
    initializations.append(p4)
    
    # Refine each initialization with local search
    for idx, p_init in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        ls_deadline = deadline - 25.0 + idx * 5.0
        p_ls, c_ls = _local_search(a, b, p_init, ls_deadline, max_passes=30)
        
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()
    
    # Phase 2: Simulated Annealing from best solution
    if time.time() < deadline - 20.0 and best_p is not None:
        sa_deadline = deadline - 15.0
        p_sa, c_sa = _sa_run(a, b, best_p.copy(), sa_deadline, initial_temp=500.0, cooling=0.97, max_iters=3000)
        
        if c_sa < best_cost:
            best_cost = c_sa
            best_p = p_sa.copy()
    
    # Phase 3: Iterated Local Search
    current_p = best_p.copy()
    current_cost = best_cost
    ils_deadline = deadline - 5.0
    temperature = 1.0
    
    iteration = 0
    while time.time() < ils_deadline:
        iteration += 1
        temperature = max(0.1, 1.0 - 0.02 * iteration)
        
        # Perturb: k random swaps
        k = 2 + int(temperature * 3)
        perturbed_p = current_p.copy()
        for _ in range(k):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]
        
        # Local search
        p_ls, c_ls = _local_search(a, b, perturbed_p, ils_deadline, max_passes=20)
        
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()
            current_p = p_ls.copy()
            current_cost = c_ls
        elif c_ls < current_cost:
            delta = c_ls - current_cost
            accept_prob = np.exp(-delta / (temperature * 500.0))
            if rng.random() < accept_prob:
                current_p = p_ls.copy()
                current_cost = c_ls
    
    # Phase 4: Final polish with more local search passes
    if time.time() < deadline - 1.0 and best_p is not None:
        final_p = best_p.copy()
        final_p, final_cost = _local_search(a, b, final_p, deadline - 0.1, max_passes=10)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
