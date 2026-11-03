"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute the QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return np.sum(a * b[np.ix_(p, p)])


def _delta_swap(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    p is a numpy array. Returns delta = cost_after - cost_before.
    O(n) instead of O(n^2).
    """
    pi = p[i]
    pj = p[j]
    
    # Cost contribution involving i or j
    # Before: for all k, a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] 
    #         for all k, a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    # After:  for all k, a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj]
    #         for all k, a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    
    # Let's compute delta more carefully.
    # Total cost C = sum_{u,v} a[u,v] b[p[u], p[v]]
    # Swap p[i] and p[j].
    # Only terms involving index i or j change.
    
    # Terms where u=i or v=i or u=j or v=j.
    
    # Let's use the standard delta formula for QAP swap.
    # delta = sum_{k != i, k != j} [ a[i,k] (b[pj, p[k]] - b[pi, p[k]]) + a[k,i] (b[p[k], pj] - b[p[k], pi])
    #                              + a[j,k] (b[pi, p[k]] - b[pj, p[k]]) + a[k,j] (b[p[k], pi] - b[p[k], pj]) ]
    #       + a[i,j] (b[pj, pi] - b[pi, pj]) + a[j,i] (b[pi, pj] - b[pj, pi])
    
    # Note: b is symmetric? In QAP, distance matrix is usually symmetric. But let's not assume.
    # Actually, for tho150, b is symmetric. a is symmetric too.
    # But let's write it generally.
    
    # Let's precompute vectors to speed up.
    # row_i = a[i, :], col_i = a[:, i]
    # row_j = a[j, :], col_j = a[:, j]
    
    # For k not in {i, j}:
    # term_i_k = a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
    # term_k_i = a[k,i] * (b[p[k], pj] - b[p[k], pi])
    # term_j_k = a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
    # term_k_j = a[k,j] * (b[p[k], pi] - b[p[k], pj])
    
    # Sum over k not i,j.
    # We can compute sum over all k and subtract k=i and k=j.
    
    # Let's compute using numpy vectorization for speed.
    # p is the current permutation.
    # p_i = p[i], p_j = p[j]
    
    # We need:
    # S1 = sum_{k != i,j} a[i,k] * b[pj, p[k]]
    # S2 = sum_{k != i,j} a[i,k] * b[pi, p[k]]
    # S3 = sum_{k != i,j} a[k,i] * b[p[k], pj]
    # S4 = sum_{k != i,j} a[k,i] * b[p[k], pi]
    # S5 = sum_{k != i,j} a[j,k] * b[pi, p[k]]
    # S6 = sum_{k != i,j} a[j,k] * b[pj, p[k]]
    # S7 = sum_{k != i,j} a[k,j] * b[p[k], pi]
    # S8 = sum_{k != i,j} a[k,j] * b[p[k], pj]
    
    # This is 8 sums. We can combine them.
    # delta = (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8) + cross_terms
    
    # Let's compute:
    # diff_b_pi_pj = b[pj, :] - b[pi, :]  # shape (n,)
    # diff_b_pj_pi = b[pi, :] - b[pj, :]  # shape (n,)
    
    # For row i:
    # sum_{k != i,j} a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
    # = sum_{k != i,j} a[i,k] * diff_b_pi_pj[p[k]]
    
    # Let's precompute:
    # v_pi = b[pi, :]  # row pi of b
    # v_pj = b[pj, :]  # row pj of b
    # v_pi_T = b[:, pi]  # col pi of b
    # v_pj_T = b[:, pj]  # col pj of b
    
    # Then:
    # term1 = sum_{k != i,j} a[i,k] * (v_pj[p[k]] - v_pi[p[k]])
    # term2 = sum_{k != i,j} a[k,i] * (v_pj_T[p[k]] - v_pi_T[p[k]])
    # term3 = sum_{k != i,j} a[j,k] * (v_pi[p[k]] - v_pj[p[k]])
    # term4 = sum_{k != i,j} a[k,j] * (v_pi_T[p[k]] - v_pj_T[p[k]])
    
    # Cross terms (k=i or k=j):
    # When k=i:
    # a[i,i]*(b[pj,pi]-b[pi,pj]) + a[i,i]*(b[pi,pj]-b[pj,pi]) = 0? No, wait.
    # The formula above sums over k != i,j. So we handle k=i and k=j separately.
    
    # Let's just compute the full delta by summing over all k and subtracting the i,j terms,
    # or better, just compute the 4 sums over all k and adjust.
    
    # Actually, let's use a simpler approach: compute the change in cost directly.
    # C_new - C_old = sum_{u,v} a[u,v] (b[p_new[u], p_new[v]] - b[p_old[u], p_old[v]])
    # Only p[i] and p[j] change.
    # p_new[i] = p_old[j], p_new[j] = p_old[i], p_new[k] = p_old[k] for k != i,j.
    
    # So delta = 
    #   a[i,i] * (b[pj,pj] - b[pi,pi])
    # + a[j,j] * (b[pi,pi] - b[pj,pj])
    # + a[i,j] * (b[pj,pi] - b[pi,pj])
    # + a[j,i] * (b[pi,pj] - b[pj,pi])
    # + sum_{k != i,j} [ a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
    #                   + a[k,i] * (b[p[k], pj] - b[p[k], pi])
    #                   + a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
    #                   + a[k,j] * (b[p[k], pi] - b[p[k], pj]) ]
    
    # Let's compute this efficiently.
    n = len(p)
    pi = p[i]
    pj = p[j]
    
    # Precompute vectors
    b_row_pi = b[pi, :]
    b_row_pj = b[pj, :]
    b_col_pi = b[:, pi]
    b_col_pj = b[:, pj]
    
    # Differences
    diff_row = b_row_pj - b_row_pi  # b[pj, :] - b[pi, :]
    diff_col = b_col_pj - b_col_pi  # b[:, pj] - b[:, pi]
    
    # Sum over all k
    # S1 = sum_k a[i,k] * diff_row[p[k]]
    # S2 = sum_k a[k,i] * diff_col[p[k]]
    # S3 = sum_k a[j,k] * (-diff_row[p[k]])  # b[pi, p[k]] - b[pj, p[k]] = -diff_row
    # S4 = sum_k a[k,j] * (-diff_col[p[k]])
    
    # S1 = a[i, :] @ diff_row[p]
    # S2 = a[:, i] @ diff_col[p]
    # S3 = -a[j, :] @ diff_row[p]
    # S4 = -a[:, j] @ diff_col[p]
    
    # But we need to exclude k=i and k=j.
    # Let's compute the full sums and subtract the k=i and k=j terms.
    
    full_S1 = np.dot(a[i, :], diff_row[p])
    full_S2 = np.dot(a[:, i], diff_col[p])
    full_S3 = -np.dot(a[j, :], diff_row[p])
    full_S4 = -np.dot(a[:, j], diff_col[p])
    
    # Subtract k=i terms:
    # For S1, k=i: a[i,i] * diff_row[p[i]] = a[i,i] * diff_row[pi]
    # For S2, k=i: a[i,i] * diff_col[p[i]] = a[i,i] * diff_col[pi]
    # For S3, k=i: -a[j,i] * diff_row[p[i]] = -a[j,i] * diff_row[pi]
    # For S4, k=i: -a[i,j] * diff_col[p[i]] = -a[i,j] * diff_col[pi]
    
    # Subtract k=j terms:
    # For S1, k=j: a[i,j] * diff_row[p[j]] = a[i,j] * diff_row[pj]
    # For S2, k=j: a[j,i] * diff_col[p[j]] = a[j,i] * diff_col[pj]
    # For S3, k=j: -a[j,j] * diff_row[p[j]] = -a[j,j] * diff_row[pj]
    # For S4, k=j: -a[j,j] * diff_col[p[j]] = -a[j,j] * diff_col[pj]
    
    sub_S1_i = a[i, i] * diff_row[pi]
    sub_S2_i = a[i, i] * diff_col[pi]
    sub_S3_i = -a[j, i] * diff_row[pi]
    sub_S4_i = -a[i, j] * diff_col[pi]
    
    sub_S1_j = a[i, j] * diff_row[pj]
    sub_S2_j = a[j, i] * diff_col[pj]
    sub_S3_j = -a[j, j] * diff_row[pj]
    sub_S4_j = -a[j, j] * diff_col[pj]
    
    sum_S1 = full_S1 - sub_S1_i - sub_S1_j
    sum_S2 = full_S2 - sub_S2_i - sub_S2_j
    sum_S3 = full_S3 - sub_S3_i - sub_S3_j
    sum_S4 = full_S4 - sub_S4_i - sub_S4_j
    
    # Cross terms (i,j and j,i)
    cross_ii = a[i, i] * (b[pj, pj] - b[pi, pi])
    cross_jj = a[j, j] * (b[pi, pi] - b[pj, pj])
    cross_ij = a[i, j] * (b[pj, pi] - b[pi, pj])
    cross_ji = a[j, i] * (b[pi, pj] - b[pj, pi])
    
    delta = sum_S1 + sum_S2 + sum_S3 + sum_S4 + cross_ii + cross_jj + cross_ij + cross_ji
    
    return delta


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Hungarian initialization
    # The QAP can be approximated by two linear assignment problems.
    # A common heuristic is to use the row sums of a and b.
    # Let's try a more sophisticated initialization: 
    # Solve a linear assignment problem where cost[i][j] = a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # This is the "rank matching" idea but optimized via Hungarian.
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Create cost matrix for Hungarian
    # cost[i][j] = a_row_sums[i] * b_row_sums[j]
    # But this is O(n^2) to create, which is fine for n=150.
    # Actually, we can use the fact that cost[i][j] = u[i] * v[j]
    # The Hungarian algorithm on such a matrix has a specific structure.
    # But let's just create it and solve.
    
    cost_mat = np.outer(a_row_sums, b_row_sums)
    
    # We want to minimize sum a[i,j] b[p[i], p[j]].
    # If we assign facility i to location j, the "cost" is roughly a_row_sums[i] * b_row_sums[j].
    # We want heavy flows (high a_row_sums) to go to central locations (low b_row_sums).
    # So we should minimize a_row_sums[i] * b_row_sums[j]? 
    # Wait, if a_row_sums[i] is large, we want b_row_sums[j] to be small.
    # So the product is small. Yes, minimizing the product is correct.
    
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Compute initial cost
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Simulated Annealing with delta swaps
    # Temperature schedule
    T_start = 1000.0
    T_end = 1.0
    cooling_rate = 0.9999  # Adjust based on time
    
    # Precompute some things for speed?
    # a and b are fixed.
    
    rng = np.random.default_rng(42)
    
    T = T_start
    while time.time() < deadline:
        # Randomly pick two distinct indices i, j
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        delta = _delta_swap(a, b, p, i, j)
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
