# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    # Time budget: 115 seconds to stay safely under 120s limit
    start_time = time.time()
    time_budget = 115.0

    # Step 1: Initial assignment using rank-matching (heavy flows to short distances)
    # This gives a reasonable starting point
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Compute initial cost
    def compute_cost(perm):
        # perm[i] is the location assigned to facility i
        # cost = sum_{i,j} a[i,j] * b[perm[i], perm[j]]
        # We can compute this efficiently
        # b_perm = b[np.ix_(perm, perm)]
        # cost = np.sum(a * b_perm)
        # But for large n, let's be careful
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)

    current_cost = compute_cost(p)

    # Step 2: Greedy 2-opt local search
    # Try swapping pairs of facilities and accept if it reduces cost
    # For efficiency, we precompute some structures

    # Precompute row sums of a for quick delta computation
    # When we swap locations of facilities i and j:
    # The change in cost involves terms where i or j are involved
    # delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    #        + sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) + sum_k a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    #        - 2*a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])  # wait, need to be careful

    # Actually, let's think more carefully.
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # If we swap p[i] and p[j], let old_i = p[i], old_j = p[j]
    # New p'[i] = old_j, p'[j] = old_i
    # The terms that change are those involving i or j in either position
    # Terms with i as first index: sum_j' a[i,j'] * b[old_j, p[j']] (was b[old_i, p[j']])
    # Terms with i as second index: sum_j' a[j',i] * b[p[j'], old_j] (was b[p[j'], old_i])
    # Terms with j as first index: sum_i' a[j,i'] * b[old_i, p[i']] (was b[old_j, p[i']])
    # Terms with j as second index: sum_i' a[i',j] * b[p[i'], old_i] (was b[p[i'], old_j])
    # But we double count (i,j) and (j,i) terms.

    # Let's define delta more carefully:
    # Old contribution from pairs involving i or j:
    #   sum_{k != i,j} a[i,k]*b[old_i, p[k]] + a[j,k]*b[old_j, p[k]] + a[k,i]*b[p[k], old_i] + a[k,j]*b[p[k], old_j]
    #   + a[i,j]*b[old_i, old_j] + a[j,i]*b[old_j, old_i]
    # New contribution:
    #   sum_{k != i,j} a[i,k]*b[old_j, p[k]] + a[j,k]*b[old_i, p[k]] + a[k,i]*b[p[k], old_j] + a[k,j]*b[p[k], old_i]
    #   + a[i,j]*b[old_j, old_i] + a[j,i]*b[old_i, old_j]
    # Delta = new - old

    # For efficiency, let's precompute:
    # For each location l, we can compute:
    #   S1[l] = sum_k a[k, l] * b[p[k], l]  -- no, this doesn't quite work directly

    # Let's just do a straightforward but vectorized 2-opt.
    # For n=100, we can afford O(n^2) swaps per iteration, each O(n) to evaluate.
    # That's O(n^3) per full pass = 10^6 operations, which is fine.

    # Precompute for fast delta evaluation
    # Let's define:
    #   A_i = a[i, :]  (row i of a)
    #   B_col_l = b[:, l]  (column l of b)
    #   B_row_l = b[l, :]  (row l of b)

    # For a swap of facilities i and j (with current locations li = p[i], lj = p[j]):
    # delta = sum_{k != i,j} [
    #   a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    # + a[j,k] * (b[li, p[k]] - b[lj, p[k]])
    # + a[k,i] * (b[p[k], lj] - b[p[k], li])
    # + a[k,j] * (b[p[k], li] - b[p[k], lj])
    # ]
    # + a[i,j]*(b[lj, li] - b[li, lj])
    # + a[j,i]*(b[li, lj] - b[lj, li])

    # Note that b[lj, li] - b[li, lj] = -(b[li, lj] - b[lj, li]), so:
    # a[i,j]*(b[lj, li] - b[li, lj]) + a[j,i]*(b[li, lj] - b[lj, li])
    # = (a[j,i] - a[i,j]) * (b[li, lj] - b[lj, li])

    # Let's precompute some vectors for speed.
    # For each facility i, precompute:
    #   row_a_i = a[i, :]
    #   col_a_i = a[:, i]

    # During 2-opt, for a candidate swap (i, j):
    # We need:
    #   diff_row = b[lj, :] - b[li, :]   (row lj minus row li of b)
    #   diff_col = b[:, lj] - b[:, li]   (col lj minus col li of b)

    # Then:
    #   sum_k a[i,k] * diff_row[p[k]]  for k != i,j
    #   sum_k a[j,k] * (-diff_row[p[k]]) for k != i,j  => -sum_k a[j,k] * diff_row[p[k])
    #   sum_k a[k,i] * diff_col[p[k]]  for k != i,j
    #   sum_k a[k,j] * (-diff_col[p[k]]) for k != i,j  => -sum_k a[k,j] * diff_col[p[k])

    # Let's precompute for each location l:
    #   For a given set of assignments, we can compute:
    #   R[l] = sum_k a[k, ?] * b[p[k], l] -- this depends on which column we're looking at

    # This is getting complex. Let's just do it directly but vectorized over k.

    # Precompute:
    #   a_rows = a  (n x n)
    #   a_cols = a.T  (n x n)
    #   b_rows = b  (n x n)
    #   b_cols = b.T  (n x n)

    # For a swap (i, j) with li = p[i], lj = p[j]:
    #   diff_row = b[lj, :] - b[li, :]   (vector of length n)
    #   diff_col = b[:, lj] - b[:, li]   (vector of length n)

    #   # Sum over k != i,j of a[i,k] * diff_row[p[k]]
    #   # = sum_k a[i,k] * diff_row[p[k]] - a[i,i]*diff_row[p[i]] - a[i,j]*diff_row[p[j]]
    #   # But p[i] = li, p[j] = lj
    #   # = sum_k a[i,k] * diff_row[p[k]] - a[i,i]*diff_row[li] - a[i,j]*diff_row[lj]

    # This is still O(n) per term. With n=100, and O(n^2) pairs, that's O(n^3) = 10^6, which is fine.

    # Let's implement a simple 2-opt with vectorized inner loop.

    best_cost = current_cost
    best_p = p.copy()

    # Precompute a and a.T for access
    a_rows = a
    a_cols = a.T
    b_matrix = b

    # Helper to compute delta for swapping i and j
    def compute_delta(i, j, perm, a_r, a_c, b_m):
        li = perm[i]
        lj = perm[j]
        if li == lj:
            return 0

        diff_row = b_m[lj, :] - b_m[li, :]  # b[lj, :] - b[li, :]
        diff_col = b_m[:, lj] - b_m[:, li]  # b[:, lj] - b[:, li]

        # Sum over k of a[i,k] * diff_row[perm[k]]
        s1 = np.dot(a_r[i, :], diff_row[perm])
        # Subtract terms for k=i and k=j
        s1 -= a_r[i, i] * diff_row[li]
        s1 -= a_r[i, j] * diff_row[lj]

        # Sum over k of a[j,k] * (-diff_row[perm[k]])
        s2 = -np.dot(a_r[j, :], diff_row[perm])
        s2 += a_r[j, i] * diff_row[li]
        s2 += a_r[j, j] * diff_row[lj]

        # Sum over k of a[k,i] * diff_col[perm[k]]
        s3 = np.dot(a_c[i, :], diff_col[perm])
        s3 -= a_c[i, i] * diff_col[li]
        s3 -= a_c[i, j] * diff_col[lj]

        # Sum over k of a[k,j] * (-diff_col[perm[k]])
        s4 = -np.dot(a_c[j, :], diff_col[perm])
        s4 += a_c[j, i] * diff_col[li]
        s4 += a_c[j, j] * diff_col[lj]

        # The (i,j) and (j,i) cross terms
        cross = (a_r[j, i] - a_r[i, j]) * (b_m[li, lj] - b_m[lj, li])

        return s1 + s2 + s3 + s4 + cross

    # 2-opt local search
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start_time > time_budget - 5:
                    # Leave some time for returning
                    break
                delta = compute_delta(i, j, p, a_rows, a_cols, b_matrix)
                if delta < -1e-9:  # Accept if it improves
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
            if time.time() - start_time > time_budget - 5:
                break

    return best_p.tolist()
# EVOLVE-BLOCK-END
