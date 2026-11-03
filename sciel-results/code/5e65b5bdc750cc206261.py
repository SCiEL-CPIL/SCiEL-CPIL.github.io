"""QAP solver for tai256c with multi-start 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _rank_matching_init(a, b):
    """Rank-matching: heavy flow rows to central location rows."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p


def _spectral_init(a, b, n):
    """Spectral embedding initial permutation."""
    # Construct a linearization matrix M
    # M[i,j] = a[i,j] * (row_sum_a[i] + col_sum_b[j]) as a heuristic
    row_sum_a = a.sum(axis=1)
    col_sum_b = b.sum(axis=0)
    # Use a simpler spectral approach: embed based on a's structure
    # Compute leading eigenvectors of a scaled matrix
    # For QAP, a common linearization is M = diag(a*1) + a ⊙ b (Hadamard)
    # But we need a matrix to embed. Let's use the row-sum weighted approach.
    
    # Alternative: use the matrix A = a * np.outer(row_sum_a, np.ones(n)) 
    # or simply use a's eigenvectors
    try:
        # Symmetrize a for spectral embedding
        a_sym = (a + a.T) / 2.0
        # Get leading eigenvectors
        vals, vecs = np.linalg.eigh(a_sym)
        # Use the first few eigenvectors to create a 2D embedding
        k = min(8, n)
        embed = vecs[:, -k:]  # largest eigenvalues
        # Project to 1D using a linear combination
        # Use the first eigenvector as primary
        scores = embed[:, -1]  # largest eigenvalue component
        p = np.argsort(scores, kind="stable")
        return p.astype(np.int64)
    except:
        return _rank_matching_init(a, b)


def _random_init(n, rng):
    """Random permutation."""
    p = np.arange(n, dtype=np.int64)
    rng.shuffle(p)
    return p


def _local_search_2opt(a, b, p, deadline, max_rounds=50):
    """2-opt local search with vectorized delta computation.
    
    Returns (improved_p, improved_cost, converged).
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute row and column sums of a for delta computation
    a_row_sum = a.sum(axis=1)  # sum_j a[i,j]
    a_col_sum = a.sum(axis=0)  # sum_j a[j,i]
    
    # For delta computation when swapping positions i and j:
    # delta = sum_k [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])]
    #        + sum_k [a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])]
    # 
    # Let's define:
    # For position i with facility f_i = p[i], the contribution to cost from row i is:
    #   row_contrib[i] = sum_k a[i,k] * b[f_i, p[k]]
    # And from column i:
    #   col_contrib[i] = sum_k a[k,i] * b[p[k], f_i]
    #
    # When we swap p[i] and p[j], the new costs are:
    #   new_row_i = sum_k a[i,k] * b[p[j], p[k]]  (but p[k] changes for k=i,j)
    #   new_col_i = sum_k a[k,i] * b[p[k], p[j]]
    # etc.
    
    # More efficient: precompute for each position i, the vector of b[p[i], :] and b[:, p[i]]
    # But that's O(n^2) memory. Instead, compute deltas on the fly with vectorization.
    
    # Let's use a different approach: for each i, compute the delta for swapping with all j > i
    # delta(i,j) = [sum_k a[i,k]*b[p[j],p[k]] + sum_k a[k,i]*b[p[k],p[j]]]
    #            + [sum_k a[j,k]*b[p[i],p[k]] + sum_k a[k,j]*b[p[k],p[i]]]
    #            - [sum_k a[i,k]*b[p[i],p[k]] + sum_k a[k,i]*b[p[k],p[i]]]
    #            - [sum_k a[j,k]*b[p[j],p[k]] + sum_k a[k,j]*b[p[k],p[j]]]
    
    # Precompute: for each position i, 
    #   R[i] = sum_k a[i,k] * b[p[i], p[k]]  (row contribution of position i)
    #   C[i] = sum_k a[k,i] * b[p[k], p[i]]  (col contribution of position i)
    #   total = sum_i (R[i] + C[i]) = cost
    
    # When swapping i and j:
    #   New R[i] = sum_k a[i,k] * b[p[j], p[k]]  but p[k] for k=i is now p[j], for k=j is p[i]
    #   This is getting complex. Let me use a direct approach.
    
    # Direct delta computation for swap (i,j):
    # The terms that change are those where i or j is involved.
    # Old: a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
    #      + sum_{k!=i,j} a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]]
    #      + sum_{k!=i,j} a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
    # New: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    #      + sum_{k!=i,j} a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]]
    #      + sum_{k!=i,j} a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
    
    # delta = [a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]] - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]]
    #       + sum_{k!=i,j} [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])]
    #       + sum_{k!=i,j} [a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])]
    
    # For efficiency, let's precompute:
    #   For each position i, define:
    #     b_row_i = b[p[i], :]  (length n)
    #     b_col_i = b[:, p[i]]  (length n)
    #   Then:
    #     R[i] = sum_k a[i,k] * b_row_i[k]
    #     C[i] = sum_k a[k,i] * b_col_i[k]
    
    # When swapping i and j:
    #   delta = [a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])]
    #         + sum_{k!=i,j} (a[i,k] - a[j,k])*(b[p[j],p[k]] - b[p[i],p[k]])
    #         + sum_{k!=i,j} (a[k,i] - a[k,j])*(b[p[k],p[j]] - b[p[k],p[i]])
    
    # This is still O(n) per pair. Let's vectorize over j for fixed i.
    
    # Precompute b_row and b_col for all positions
    b_rows = b[p, :]  # shape (n, n), b_rows[i] = b[p[i], :]
    b_cols = b[:, p]  # shape (n, n), b_cols[i] = b[:, p[i]]
    
    # R[i] = sum_k a[i,k] * b_rows[i,k]
    # C[i] = sum_k a[k,i] * b_cols[k,i]
    
    # For fixed i, compute delta for all j:
    # delta(i,j) = a[i,j]*(b_rows[j,i] - b_rows[i,j]) + a[j,i]*(b_rows[i,j] - b_rows[j,i])
    #            + sum_{k!=i,j} (a[i,k] - a[j,k])*(b_rows[j,k] - b_rows[i,k])
    #            + sum_{k!=i,j} (a.T[i,k] - a.T[j,k])*(b_cols[k,j] - b_cols[k,i])
    
    # Note: b_rows[j,i] = b[p[j], p[i]], b_rows[i,j] = b[p[i], p[j]]
    
    # Let's simplify:
    # term1(i,j) = (a[i,j] - a[j,i]) * (b_rows[j,i] - b_rows[i,j])
    # term2(i,j) = sum_{k!=i,j} (a[i,k] - a[j,k]) * (b_rows[j,k] - b_rows[i,k])
    # term3(i,j) = sum_{k!=i,j} (a.T[i,k] - a.T[j,k]) * (b_cols[k,j] - b_cols[k,i])
    
    # For term2: sum_{k!=i,j} (a[i,k] - a[j,k]) * (b_rows[j,k] - b_rows[i,k])
    # = sum_k (a[i,k] - a[j,k]) * (b_rows[j,k] - b_rows[i,k])
    #   - (a[i,i] - a[j,i]) * (b_rows[j,i] - b_rows[i,i])
    #   - (a[i,j] - a[j,j]) * (b_rows[j,j] - b_rows[i,j])
    
    # This is getting complex. Let me just do a straightforward O(n^2) per round with vectorization.
    
    converged = False
    round_num = 0
    
    while not converged and round_num < max_rounds:
        round_num += 1
        if time.time() > deadline:
            break
        
        improved = False
        # Try all pairs (i, j) with i < j
        # For each i, compute deltas for all j > i in a vectorized manner
        for i in range(n):
            if time.time() > deadline:
                return p, current_cost, False
            
            # Compute delta for swapping i with all j > i
            # Using the formula:
            # delta(i,j) = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
            #            + sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
            #            + sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
            
            # Vectorized computation for all j:
            # Let's define:
            # diff_a_row = a[i, :] - a[:, i].T  # shape (n, n), diff_a_row[i,j] = a[i,j] - a[j,i]
            # But we need (a[i,k] - a[j,k]) for all j,k, which is a[i,:] - a[j,:]
            
            # For fixed i, compute:
            # b_diff = b_rows[j, :] - b_rows[i, :] for all j  -> shape (n, n)
            # a_diff = a[i, :] - a[j, :] for all j  -> shape (n, n)
            # term2 = sum_k a_diff[j,k] * b_diff[j,k] for each j
            
            # b_rows has shape (n,n), b_rows[i] = b[p[i],:]
            # b_diff[j, k] = b_rows[j, k] - b_rows[i, k]
            b_diff = b_rows - b_rows[i]  # shape (n, n), b_diff[j,k] = b[p[j],p[k]] - b[p[i],p[k]]
            
            # a_diff[j, k] = a[i,k] - a[j,k]
            a_diff = a[i] - a  # shape (n, n), a_diff[j,k] = a[i,k] - a[j,k]
            
            # term2[j] = sum_k a_diff[j,k] * b_diff[j,k]
            term2 = np.sum(a_diff * b_diff, axis=1)  # shape (n,)
            
            # term3: sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
            # a_T_diff[j, k] = a[k,i] - a[k,j] = a.T[i,k] - a.T[j,k]
            a_T = a.T
            a_T_diff = a_T[i] - a_T  # shape (n, n), a_T_diff[j,k] = a[k,i] - a[k,j]
            
            # b_cols[k, j] = b[p[k], p[j]], b_cols[k, i] = b[p[k], p[i]]
            # b_cols_diff[j, k] = b_cols[k, j] - b_cols[k, i]
            b_cols_diff = b_cols.T - b_cols[i]  # shape (n, n), b_cols_diff[j,k] = b[p[k],p[j]] - b[p[k],p[i]]
            
            term3 = np.sum(a_T_diff * b_cols_diff, axis=1)  # shape (n,)
            
            # term1[j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
            # = (a[i,j] - a[j,i]) * (b_rows[j,i] - b_rows[i,j])
            term1 = (a[i] - a.T[i]) * (b_rows[:, i] - b_rows[i, :])  # shape (n,)
            
            # Total delta
            delta = term1 + term2 + term3  # shape (n,)
            
            # We only consider j > i
            # Find the best j > i with delta < 0
            for j in range(i + 1, n):
                if delta[j] < -1e-9:  # significant improvement
                    # Perform the swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += int(delta[j])
                    
                    # Update b_rows and b_cols
                    b_rows[i], b_rows[j] = b_rows[j], b_rows[i]
                    b_cols[:, i], b_cols[:, j] = b_cols[:, j].copy(), b_cols[:, i].copy()
                    # Actually b_cols[k, i] = b[p[k], p[i]], so we need to update column i and j
                    # b_cols[k, i] = b[p[k], p[i]], after swap p[i] and p[j] are swapped
                    # So b_cols[:, i] should now be b[:, p_new[i]] = b[:, p_old[j]]
                    # and b_cols[:, j] should now be b[:, p_new[j]] = b[:, p_old[i]]
                    # But b_cols was defined as b[:, p], so b_cols[k, pos] = b[p[k], p[pos]]
                    # After swapping p[i] and p[j], we need:
                    # b_cols_new[k, i] = b[p_new[k], p_new[i]]
                    # For k != i,j: p_new[k] = p_old[k], p_new[i] = p_old[j]
                    # So b_cols_new[k, i] = b[p_old[k], p_old[j]] = b_cols_old[k, j]
                    # Similarly b_cols_new[k, j] = b_cols_old[k, i]
                    # For k = i: b_cols_new[i, i] = b[p_new[i], p_new[i]] = b[p_old[j], p_old[j]] = b_cols_old[j, j]
                    # For k = j: b_cols_new[j, j] = b[p_new[j], p_new[j]] = b[p_old[i], p_old[i]] = b_cols_old[i, i]
                    # So we can just swap columns i and j of b_cols
                    b_cols[:, [i, j]] = b_cols[:, [j, i]]
                    
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            converged = True
    
    return p, current_cost, converged


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Rank matching
    starts.append(_rank_matching_init(a, b))
    
    # 2. Spectral initialization
    starts.append(_spectral_init(a, b, n))
    
    # 3. Random permutations
    num_random = 5
    for _ in range(num_random):
        starts.append(_random_init(n, rng))
    
    # Evaluate all starts and find the best
    best_p = None
    best_cost = np.inf
    
    for p in starts:
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Run local search on the best start
    best_p, best_cost, _ = _local_search_2opt(a, b, best_p, deadline)
    
    # If time remains, try local search on other starts
    for p in starts[1:]:
        if time.time() > deadline:
            break
        cost = _qap_cost(a, b, p)
        if cost >= best_cost:
            continue
        p_ls, cost_ls, _ = _local_search_2opt(a, b, p, deadline)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Final polish: one more round of local search if time permits
    if time.time() < deadline - 5.0:
        best_p, best_cost, _ = _local_search_2opt(a, b, best_p, deadline)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
