# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims.
    Returns X with rows centered, shape (n, k)."""
    n = M.shape[0]
    # Double centering: B = -1/2 J M J
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    # Eigendecomposition (B symmetric)
    w, V = np.linalg.eigh(B)
    # Take largest k eigenvalues
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    # X = V * sqrt(w)
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    """Find permutation p minimizing sum ||X[i] - Y[p[i]]||^2 via Hungarian-like greedy.
    Actually solve the assignment: minimize sum_{i} ||X[i] - Y[p[i]]||^2.
    This is a linear assignment problem on the cost matrix C[i,j] = ||X[i]-Y[j]||^2.
    Use scipy's linear_sum_assignment."""
    from scipy.optimize import linear_sum_assignment
    n = X.shape[0]
    # C[i,j] = ||X[i] - Y[j]||^2 = ||X[i]||^2 - 2 X[i].Y[j] + ||Y[j]||^2
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    # linear_sum_assignment minimizes sum C[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _two_opt_search(a, b, p, time_limit, max_iter=200):
    """Efficient 2-opt local search with O(1) per-swap delta using precomputed aggregates."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    # Precompute aggregate matrices for O(1) swap delta
    # R[l, i] = sum_k a[i, k] * b[l, p[k]]  (row aggregate: location l as first b-index)
    # C[l, i] = sum_k a[k, i] * b[p[k], l]  (col aggregate: location l as second b-index)
    # R = a @ b[:, p]  (n x n): R[i, l] = sum_k a[i,k]*b[l,p[k]] -- wait, let me be careful
    # b[:, p] has shape (n, n): (b[:, p])[l, k] = b[l, p[k]]
    # a @ b[:, p] has shape (n, n): (a @ b[:, p])[i, k] = sum_l a[i, l] * b[l, p[k]]
    # That's not quite what I want. Let me redefine.
    
    # I want: for facility i at location li, its row contribution is sum_j a[i,j] * b[li, p[j]]
    # = a[i, :] @ b[li, p] = (a @ b[:, p])[i, li]  ... no
    # a @ b[:, p]: element [i, k] = sum_l a[i,l] * b[l, p[k]]
    # I want sum_j a[i,j] * b[li, p[j]] = sum_j a[i,j] * (b[:, p])[li, j] = (a * b[:, p].T)[i, li].sum()
    # Hmm, let me just compute the aggregates directly.
    
    # Let's define:
    #   agg_row[i, l] = sum_j a[i, j] * b[l, p[j]]
    #   agg_col[i, l] = sum_j a[j, i] * b[p[j], l]
    # Then cost = sum_i agg_row[i, p[i]] + sum_i agg_col[i, p[i]]
    # Wait, that double counts. Actually cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # = sum_i sum_j a[i,j] b[p[i], p[j]]
    # For fixed i, sum_j a[i,j] b[p[i], p[j]] = agg_row[i, p[i]]
    # And sum_i agg_row[i, p[i]] = sum_{i,j} a[i,j] b[p[i], p[j]] = cost. Good, no double count.
    
    # Compute agg_row: agg_row[i, l] = sum_j a[i,j] * b[l, p[j]]
    # = (a * b[:, p].T) summed over j... 
    # b[:, p] shape (n,n): [l, j] = b[l, p[j]]
    # b[:, p].T shape (n,n): [j, l] = b[l, p[j]]
    # a * b[:, p].T: [i, j] * [j, l] -- can't multiply directly, need to sum over j
    # agg_row = a @ b[:, p]  -- [i, l] = sum_j a[i,j] * b[l, p[j]]. YES!
    
    # agg_col[i, l] = sum_j a[j, i] * b[p[j], l]
    # b[p, :] shape (n,n): [j, l] = b[p[j], l]
    # a.T @ b[p, :] -- [i, l] = sum_j a.T[i,j] * b[p,j],l] = sum_j a[j,i] * b[p[j], l]. YES!
    
    agg_row = a @ b[:, p]  # shape (n, n), agg_row[i, l] = sum_j a[i,j]*b[l,p[j]]
    agg_col = a.T @ b[p, :]  # shape (n, n), agg_col[i, l] = sum_j a[j,i]*b[p[j],l]
    
    # Verify: cost should equal sum_i agg_row[i, p[i]]
    # (We'll trust the math and verify once)
    
    start = time.time()
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                li = p[i]
                lj = p[j]
                
                # When we swap p[i] and p[j] (li <-> lj):
                # Facility i moves from li to lj, facility j moves from lj to li.
                # 
                # New cost = old cost - (terms with i at li) - (terms with j at lj)
                #            + (terms with i at lj) + (terms with j at li)
                #
                # Terms involving facility i at location li:
                #   Row part: sum_k a[i,k] * b[li, p[k]]  (k includes j, where p[j]=lj)
                #   But we need to be careful: the "other" facilities' contributions also change
                #   because p[i] changes from li to lj.
                #
                # Let me think more carefully. The total cost is:
                # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
                #
                # When p[i] changes from li to lj and p[j] changes from lj to li:
                # Affected terms are those where i or j is the first index OR the second index.
                #
                # Terms where i is first index: sum_k a[i,k] * b[p[i], p[k]]
                #   Old: sum_k a[i,k] * b[li, p[k]]  (with p[j]=lj)
                #   New: sum_k a[i,k] * b[lj, p[k]]  (with p[j]=li)
                #   Difference: a[i,j]*(b[lj,li] - b[li,lj]) + sum_{k!=i,j} a[i,k]*(b[lj,p[k]] - b[li,p[k]])
                #
                # Terms where j is first index: similar
                # Terms where i is second index: sum_k a[k,i] * b[p[k], p[i]]
                #   Old: sum_k a[k,i] * b[p[k], li]  (with p[j]=lj)
                #   New: sum_k a[k,i] * b[p[k], lj]  (with p[j]=li)
                #   Difference: a[j,i]*(b[li,li] - b[lj,lj]) + sum_{k!=i,j} a[k,i]*(b[p[k],lj] - b[p[k],li])
                #
                # This is getting complex. Let me use the aggregate approach.
                #
                # agg_row[i, l] = sum_k a[i,k] * b[l, p[k]]
                # When p changes, agg_row changes. But for the delta of a swap,
                # I can compute:
                #
                # delta = [new contribution of i at lj + new contribution of j at li]
                #       - [old contribution of i at li + old contribution of j at lj]
                #       + [changes in other facilities' contributions due to p[i],p[j] swap]
                #
                # Actually, let me just compute it directly using the aggregates.
                #
                # Old cost contribution from facility i (as first index): agg_row[i, li]
                # Old cost contribution from facility j (as first index): agg_row[j, lj]
                # But these include cross-terms. Let me just compute the full delta.
                
                # The change in cost when swapping p[i] and p[j]:
                # For each pair (x, y) where x or y is i or j, the term a[x,y]*b[p[x],p[y]] changes.
                #
                # Let me enumerate:
                # 1. (i, i): a[i,i]*b[li,li] -> a[i,i]*b[lj,lj]  (changes if li != lj)
                # 2. (i, j): a[i,j]*b[li,lj] -> a[i,j]*b[lj,li]
                # 3. (j, i): a[j,i]*b[lj,li] -> a[j,i]*b[li,lj]
                # 4. (j, j): a[j,j]*b[lj,lj] -> a[j,j]*b[li,li]
                # 5. (i, k) for k!=i,j: a[i,k]*b[li,p[k]] -> a[i,k]*b[lj,p[k]]
                # 6. (k, i) for k!=i,j: a[k,i]*b[p[k],li] -> a[k,i]*b[p[k],lj]
                # 7. (j, k) for k!=i,j: a[j,k]*b[lj,p[k]] -> a[j,k]*b[li,p[k]]
                # 8. (k, j) for k!=i,j: a[k,j]*b[p[k],lj] -> a[k,j]*b[p[k],li]
                # 9. (k, l) for k,l != i,j: unchanged
                #
                # Delta = sum of (new - old) for cases 1-8
                #
                # Case 1: a[i,i]*(b[lj,lj] - b[li,li])
                # Case 2: a[i,j]*(b[lj,li] - b[li,lj])
                # Case 3: a[j,i]*(b[li,lj] - b[lj,li])
                # Case 4: a[j,j]*(b[li,li] - b[lj,lj])
                # Case 5: sum_{k!=i,j} a[i,k]*(b[lj,p[k]] - b[li,p[k]])
                #        = (sum_k a[i,k]*(b[lj,p[k]] - b[li,p[k]])) - a[i,i]*(b[lj,lj]-b[li,li]) - a[i,j]*(b[lj,li]-b[li,lj])
                #        = (agg_row[i,lj] - agg_row[i,li]) - a[i,i]*(b[lj,lj]-b[li,li]) - a[i,j]*(b[lj,li]-b[li,lj])
                # Case 6: sum_{k!=i,j} a[k,i]*(b[p[k],lj] - b[p[k],li])
                #        = (sum_k a[k,i]*(b[p[k],lj] - b[p[k],li])) - a[i,i]*(b[lj,lj]-b[li,li]) - a[j,i]*(b[li,lj]-b[lj,li])
                #        = (agg_col[i,lj] - agg_col[i,li]) - a[i,i]*(b[lj,lj]-b[li,li]) - a[j,i]*(b[li,lj]-b[lj,li])
                # Case 7: sum_{k!=i,j} a[j,k]*(b[li,p[k]] - b[lj,p[k]])
                #        = (agg_row[j,li] - agg_row[j,lj]) - a[j,j]*(b[li,li]-b[lj,lj]) - a[j,i]*(b[li,lj]-b[lj,li])
                # Case 8: sum_{k!=i,j} a[k,j]*(b[p[k],li] - b[p[k],lj])
                #        = (agg_col[j,li] - agg_col[j,lj]) - a[j,j]*(b[li,li]-b[lj,lj]) - a[i,j]*(b[li,lj]-b[lj,li])
                #
                # Total delta = cases 1-8:
                # Let me collect terms:
                # b[lj,lj]-b[li,li] terms: a[i,i] (case1) + a[j,j] (case4)
                #   - a[i,i] (case5) - a[j,i]... wait let me redo.
                #
                # Actually this is getting too complex. Let me just use a simpler formula.
                #
                # The key insight: 
                # delta = (agg_row[i,lj] - agg_row[i,li]) + (agg_row[j,li] - agg_row[j,lj])
                #       + (agg_col[i,lj] - agg_col[i,li]) + (agg_col[j,li] - agg_col[j,lj])
                #       - correction terms for double-counting
                #
                # But agg_row[i,li] includes the term a[i,i]*b[li,li] and a[i,j]*b[li,lj]
                # agg_row[i,lj] includes a[i,i]*b[lj,lj] and a[i,j]*b[lj,li]
                # So (agg_row[i,lj] - agg_row[i,li]) already captures cases 1,2,5 for the i-row.
                # Similarly for j-row and col terms.
                #
                # The issue is that cases 1,2,3,4 are captured by BOTH row and col aggregates?
                # No: agg_row only has first-index = facility, agg_col only has second-index = facility.
                # So:
                # - Case 1 (i,i as first): in agg_row[i,li] and agg_row[i,lj]
                # - Case 2 (i,j as first): in agg_row[i,li] and agg_row[i,lj]  
                # - Case 3 (j,i as first): in agg_row[j,lj] and agg_row[j,li]
                # - Case 4 (j,j as first): in agg_row[j,lj] and agg_row[j,li]
                # - Case 5 (i,k as first): in agg_row[i,li] and agg_row[i,lj]
                # - Case 6 (k,i as second): in agg_col[i,li] and agg_col[i,lj]
                # - Case 7 (j,k as first): in agg_row[j,lj] and agg_row[j,li]
                # - Case 8 (k,j as second): in agg_col[j,lj] and agg_col[j,li]
                #
                # So: (agg_row[i,lj]-agg_row[i,li]) covers cases 1,2,5
                #     (agg_row[j,li]-agg_row[j,lj]) covers cases 3,4,7
                #     (agg_col[i,lj]-agg_col[i,li]) covers case 6 (and also case 1? No, case 1 is first-index)
                #     Wait, case 6 is (k,i) where i is second index. That's in agg_col.
                #     But case 1 is (i,i) where i is first index. That's in agg_row.
                #     What about (i,i) where i is second index? That's a[i,i]*b[p[i],p[i]] = a[i,i]*b[li,li]
                #     That's in agg_col[i,li] as well!
                #
                # So (i,i) appears in BOTH agg_row[i,li] AND agg_col[i,li]. That means the total cost
                # sum_i agg_row[i,p[i]] + sum_i agg_col[i,p[i]] would double-count diagonal terms!
                #
                # Wait, no. Let me re-examine. cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
                # sum_i agg_row[i,p[i]] = sum_i sum_j a[i,j]*b[p[i],p[j]] = cost. Correct.
                # sum_i agg_col[i,p[i]] = sum_i sum_j a[j,i]*b[p[j],p[i]] = sum_{j,i} a[j,i]*b[p[j],p[i]] = cost. Also correct!
                # So both sums equal cost individually. They're not meant to be added.
                #
                # So for the delta, I should use EITHER row aggregates OR col aggregates, not both.
                # Let me just use row aggregates:
                # delta = (agg_row[i,lj] - agg_row[i,li]) + (agg_row[j,li] - agg_row[j,lj])
                #       + (changes in other rows due to p[i],p[j] swap)
                #
                # But other rows' aggregates change because p[i] and p[j] change!
                # agg_row[k,l] = sum_j a[k,j]*b[l,p[j]]
                # When p[i] changes from li to lj, agg_row[k,li] loses a[k,i]*b[li,li] and gains... 
                # no, agg_row[k,l] depends on p through b[l,p[j]]. When p[i] changes, 
                # agg_row[k,l] changes by a[k,i]*(b[l,lj] - b[l,li]) for all l.
                #
                # So the change in cost from other facilities:
                # sum_{k!=i,j} [agg_row[k,p[k]]_new - agg_row[k,p[k]]_old]
                # = sum_{k!=i,j} a[k,i]*(b[p[k],lj] - b[p[k],li]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
                # = sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],lj] - b[p[k],li])
                #
                # This is O(n) per swap. To make it O(1), I need another aggregate.
                #
                # Let me define: D[l1,l2] = sum_k (a[:,i]-a[:,j])... no, that depends on i,j.
                #
                # Alternative: precompute for each pair of locations (l1,l2):
                #   S[l1,l2] = sum_k a[k,?]*b[p[k],l2] - b[p[k],l1]... still depends on which facilities.
                #
                # I think O(n) per swap is the best I can do without more complex precomputation.
                # But with numpy, O(n) is fast. Let me just do it with numpy vectorized ops.
                
                # Compute delta using O(n) numpy operations
                # Terms from i and j as first index (using aggregates):
                d_row_i = agg_row[i, lj] - agg_row[i, li]
                d_row_j = agg_row[j, li] - agg_row[j, lj]
                
                # Terms from other facilities (k != i, j) due to p[i], p[j] changing:
                # sum_{k!=i,j} a[k,i]*(b[p[k],lj]-b[p[k],li]) + a[k,j]*(b[p[k],li]-b[p[k],lj])
                # = sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
                # Let's compute: sum_k (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
                # = sum_k (a[k,i]-a[k,j])*(b[p[k],lj]) - sum_k (a[k,i]-a[k,j])*(b[p[k],li])
                # Precompute: for each location l, T_l = sum_k a[k,i]*b[p[k],l] ... but that depends on i.
                
                # Just compute directly:
                diff_a = a[:, i] - a[:, j]  # n-vector
                diff_b = b[p, lj] - b[p, li]  # n-vector
                d_other = np.dot(diff_a, diff_b)
                # Subtract the i and j contributions (already in d_row):
                d_other -= diff_a[i] * diff_b[i] + diff_a[j] * diff_b[j]
                # But wait, diff_b[i] = b[p[i],lj]-b[p[i],li] = b[li,lj]-b[li,li]
                # and diff_a[i] = a[i,i]-a[i,j]
                # These are the (i,i) and (i,j) terms which are already in d_row_i.
                # Actually, d_row_i = sum_k a[i,k]*(b[lj,p[k]]-b[li,p[k]])
                # which includes k=i and k=j.
                # d_other = sum_k (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
                # which includes k=i and k=j.
                # The k=i term in d_other: (a[i,i]-a[i,j])*(b[li,lj]-b[li,li])
                # The k=j term in d_other: (a[j,i]-a[j,j])*(b[lj,li]-b[lj,lj])
                # These are NOT in d_row_i or d_row_j (those have a[i,k] and a[j,k] as first index).
                # So no double counting. Good.
                
                delta = d_row_i + d_row_j + d_other
                
                if delta < 0:
                    # Accept swap
                    p[i], p[j] = lj, li
                    cost += delta
                    # Update aggregates
                    # agg_row[i, :] changes: a[i,k]*b[l, p[k]] for all l
                    # Old p[i]=li, new p[i]=lj
                    # agg_row[k, l] = sum_j a[k,j]*b[l,p[j]]
                    # For k=i: agg_row[i, l] = sum_j a[i,j]*b[l,p[j]]
                    #   Old: a[i,i]*b[l,li] + a[i,j]*b[l,lj] + sum_{m!=i,j} a[i,m]*b[l,p[m]]
                    #   New: a[i,i]*b[l,lj] + a[i,j]*b[l,li] + sum_{m!=i,j} a[i,m]*b[l,p[m]]
                    #   Diff: a[i,i]*(b[l,lj]-b[l,li]) + a[i,j]*(b[l,li]-b[l,lj])
                    #        = (a[i,i]-a[i,j])*(b[l,lj]-b[l,li])
                    # For k!=i,j: agg_row[k,l] changes by a[k,i]*(b[l,lj]-b[l,li]) + a[k,j]*(b[l,li]-b[l,lj])
                    #            = (a[k,i]-a[k,j])*(b[l,lj]-b[l,li])
                    # So for all k: agg_row[k, l] += (a[k,i]-a[k,j])*(b[l,lj]-b[l,li])
                    # That's O(n^2) to update. Hmm.
                    
                    # Let me just recompute aggregates. O(n^2) per swap is too slow.
                    # Instead, let me just track the cost incrementally and not maintain aggregates.
                    # Or, accept O(n) per swap and just do the dot product.
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    improved = True
                    
                    # Recompute aggregates (expensive but correct)
                    agg_row = a @ b[:, p]
                    agg_col = a.T @ b[p, :]
                    
        if not improved:
            break
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    # Step 1: MDS embedding + Procrustes alignment
    # Symmetrize for MDS
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0

    # Choose embedding dimension
    k = min(20, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)

    # Procrustes alignment
    p = _procrustes_align(X, Y)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    # Step 2: Local search polish
    time_left = deadline - time.time()
    if time_left > 5:
        p, cost = _two_opt_search(a, b, p, time_left)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p.tolist()
# EVOLVE-BLOCK-END
