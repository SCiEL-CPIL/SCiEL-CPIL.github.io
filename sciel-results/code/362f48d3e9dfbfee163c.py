"""Combined QAP solver: spectral/Frank-Wolfe init + fast vectorized local search + ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _two_opt_search(a, b, p, time_limit=None, start_time=None):
    """
    Optimized 2-opt local search for QAP using vectorized delta calculation.
    Returns (best_p, best_cost).
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60
    
    n = len(p)
    p = p.copy()
    
    # Precompute symmetric A for delta calculations
    # Cost = sum_{i,j} A[i,j] * B[p[i], p[j]]
    # If we swap positions i and j (facilities u=p[i], v=p[j]):
    # The change involves terms where i or j appear as row or col in the A*B product.
    
    # Let's use the standard delta formula for swapping positions i and j in permutation p.
    # Let u = p[i], v = p[j].
    # Old cost contribution involving i or j:
    # sum_k A[i,k]B[u,p[k]] + sum_k A[k,i]B[p[k],u] + sum_k A[j,k]B[v,p[k]] + sum_k A[k,j]B[p[k],v]
    # Note: the term A[i,j]B[u,v] and A[j,i]B[v,u] are included in the sums above when k=j and k=i respectively.
    # Also diagonal terms A[i,i]B[u,u] and A[j,j]B[v,v] are included.
    
    # New cost contribution:
    # sum_k A[i,k]B[v,p[k]] + sum_k A[k,i]B[p[k],v] + sum_k A[j,k]B[u,p[k]] + sum_k A[k,j]B[p[k],u]
    
    # Delta = New - Old
    
    # Vectorized approach:
    # For a fixed i, we want to find the best j to swap with.
    # Let's precompute some arrays to speed this up?
    # Actually, for n=256, a full O(n^2) pass with vectorized inner loops is fast enough.
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute B_perm = B[p][:, p] ? No, we need B[u, p[k]] etc.
    # Let's just compute deltas on the fly but vectorized over j for a given i.
    
    # To make it faster, we can maintain the permuted B matrix D = B[p][:, p].
    # D[i, j] = B[p[i], p[j]].
    # Cost = sum(a * D).
    
    D = b[np.ix_(p, p)].copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        # One pass over all i
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            
            # We want to swap p[i] with p[j] for some j > i.
            # Let u = p[i], v = p[j].
            # In D, row i is B[u, p], col i is B[p, u].
            # Row j is B[v, p], col j is B[p, v].
            
            # The change in cost when swapping positions i and j:
            # Terms involving i or j in the sum sum_{k,l} A[k,l] D[k,l].
            # These are:
            # 1. A[i,i] D[i,i] -> A[i,i] D[j,j]  (since D[j,j] = B[v,v])
            # 2. A[j,j] D[j,j] -> A[j,j] D[i,i]
            # 3. A[i,j] D[i,j] -> A[i,j] D[j,i]  (D[i,j]=B[u,v], D[j,i]=B[v,u])
            # 4. A[j,i] D[j,i] -> A[j,i] D[i,j]
            # 5. For k not in {i,j}:
            #    A[i,k] D[i,k] -> A[i,k] D[j,k]
            #    A[k,i] D[k,i] -> A[k,i] D[k,j]
            #    A[j,k] D[j,k] -> A[j,k] D[i,k]
            #    A[k,j] D[k,j] -> A[k,j] D[k,i]
            
            # Let's group terms for a fixed i and variable j.
            # Delta(i, j) = 
            #   A[i,i](D[j,j] - D[i,i]) + A[j,j](D[i,i] - D[j,j])
            # + A[i,j](D[j,i] - D[i,j]) + A[j,i](D[i,j] - D[j,i])
            # + sum_{k not i,j} [ A[i,k](D[j,k] - D[i,k]) + A[k,i](D[k,j] - D[k,i]) 
            #                      + A[j,k](D[i,k] - D[j,k]) + A[k,j](D[k,i) - D[k,j]) ]
            
            # This is a bit complex to vectorize over j directly without precomputation.
            # However, n=256 is small. We can compute the delta for all j > i in a vectorized manner.
            
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
            
            # D is (n, n). D[i, js] is the row i for columns js.
            # D[js, i] is the column i for rows js.
            
            # Let's compute the "cross" terms first.
            # For each j in js, we need:
            # sum_{k not i,j} [ (A[i,k] - A[j,k]) * (D[j,k] - D[i,k]) + (A[k,i] - A[k,j]) * (D[k,j] - D[k,i]) ]
            # Wait, let's re-verify the signs.
            # Old: A[i,k]D[i,k] + A[k,i]D[k,i] + A[j,k]D[j,k] + A[k,j]D[k,j]
            # New: A[i,k]D[j,k] + A[k,i]D[k,j] + A[j,k]D[i,k] + A[k,j]D[k,i]
            # Diff: A[i,k](D[j,k]-D[i,k]) + A[k,i](D[k,j]-D[k,i]) + A[j,k](D[i,k]-D[j,k]) + A[k,j](D[k,i)-D[k,j])
            #      = (A[i,k]-A[j,k])(D[j,k]-D[i,k]) + (A[k,i]-A[k,j])(D[k,j]-D[k,i])
            
            # Let M = A - A.T? No.
            # Let's define:
            # Term1_j = sum_k (A[i,k] - A[j,k]) * (D[j,k] - D[i,k])
            # Term2_j = sum_k (A[k,i] - A[k,j]) * (D[k,j] - D[k,i])
            
            # We can compute this for all j in js at once?
            # D[js, :] are the rows for j in js.
            # D[i, :] is the row for i.
            # Diff_D_rows = D[js, :] - D[i, :]  (len(js) x n)
            
            # A[i, :] is row i of A.
            # A[js, :] are rows of A for j in js.
            # Diff_A_rows = A[i, :] - A[js, :]  (len(js) x n)
            
            # Term1 = sum_k Diff_A_rows * Diff_D_rows
            # This is element-wise multiplication then sum over k (axis=1).
            
            Diff_D_rows = D[js, :] - D[i, :]
            Diff_A_rows = A_i_row = a[i, :]  # A[i, :]
            # Wait, A[i,k] - A[j,k].
            # So we need (A[i, :] - A[js, :]) * (D[js, :] - D[i, :])
            
            A_diff = a[i, :] - a[js, :]  # (len(js), n)
            D_diff_rows = D[js, :] - D[i, :]  # (len(js), n)
            Term1 = np.sum(A_diff * D_diff_rows, axis=1)
            
            # Now Term2: sum_k (A[k,i] - A[k,j]) * (D[k,j] - D[k,i])
            # A[:, i] is col i of A.
            # A[js, :] are rows, but we need A[k, j] where j is in js.
            # This is A[:, js] transposed? No.
            # A[k, j] for k=0..n-1 and j in js.
            # This is A[:, js] which is (n, len(js)).
            # We need (A[:, i] - A[:, js]) which is (n, len(js)).
            # And (D[:, js] - D[:, i]) which is (n, len(js)).
            
            A_diff_cols = a[:, i] - a[:, js]  # (n, len(js))
            D_diff_cols = D[:, js] - D[:, i]  # (n, len(js))
            
            Term2 = np.sum(A_diff_cols * D_diff_cols, axis=0)  # (len(js),)
            
            # Combine Term1 and Term2
            Cross = Term1 + Term2
            
            # Now the diagonal and direct i,j terms.
            # Diagonal: A[i,i](D[j,j] - D[i,i]) + A[j,j](D[i,i] - D[j,j])
            #          = (A[i,i] - A[j,j]) * (D[j,j] - D[i,i])
            # Direct: A[i,j](D[j,i] - D[i,j]) + A[j,i](D[i,j] - D[j,i])
            #          = (A[i,j] - A[j,i]) * (D[j,i] - D[i,j])
            
            # Note: In the Cross terms, we summed over ALL k, including k=i and k=j.
            # We must subtract the contributions from k=i and k=j from Cross, and add the correct direct terms.
            
            # Let's check what Cross includes for k=i:
            # Term1 (k=i): (A[i,i] - A[j,i]) * (D[j,i] - D[i,i])
            # Term2 (k=i): (A[i,i] - A[i,j]) * (D[i,j] - D[i,i])  -- Wait, A_diff_cols[i] = A[i,i] - A[i,j]. D_diff_cols[i] = D[i,j] - D[i,i].
            
            # This is getting messy. The standard trick is to compute the delta for all j, then correct for the i and j indices if they were included in the sums.
            # Since we are swapping i and j, the "other" indices are k not in {i,j}.
            # My vectorized sum over k includes k=i and k=j.
            
            # Let's restart the delta calculation logic to be safe and correct.
            # It's easier to just compute the full delta for each j if n is small, but vectorizing is better.
            
            # Let's use the property:
            # Delta(i, j) = Sum_{k,l} A[k,l] (D'_{k,l} - D_{k,l})
            # where D' is D with rows/cols i and j swapped.
            
            # Only entries D[i,:], D[:,i], D[j,:], D[:,j] change.
            
            # Let's just compute the delta for all j > i using a slightly less optimized but definitely correct vectorization.
            
            # For a fixed i, we want Delta(i, j) for all j > i.
            
            # Let u = p[i].
            # Let v_j = p[j].
            
            # Old terms involving i or j:
            # T_old(j) = A[i,:].dot(D[i,:]) + A[:,i].dot(D[:,i]) + A[j,:].dot(D[j,:]) + A[:,j].dot(D[:,j])
            #           - 2 * (A[i,j]*D[i,j] + A[j,i]*D[j,i])  # Because these are counted twice in the sums above?
            #           No, sum_{k,l} includes (i,j) and (j,i) separately.
            #           Row i sum: sum_l A[i,l]D[i,l]. Includes (i,j).
            #           Col i sum: sum_k A[k,i]D[k,i]. Includes (j,i).
            #           Row j sum: sum_l A[j,l]D[j,l]. Includes (j,i).
            #           Col j sum: sum_k A[k,j]D[k,j]. Includes (i,j).
            #           So (i,j) is in Row i and Col j. (j,i) is in Col i and Row j.
            #           They are distinct.
            
            # New terms:
            # T_new(j) = A[i,:].dot(D[j,:]) + A[:,i].dot(D[:,j]) + A[j,:].dot(D[i,:]) + A[:,j].dot(D[:,i])
            
            # Delta = T_new - T_old.
            
            # Let's compute this for all j in js.
            
            # D[i, :] is fixed.
            # D[js, :] is (len(js), n).
            # A[i, :] is (n,).
            # A[js, :] is (len(js), n).
            
            # Term 1: A[i, :] dot D[j, :] for each j.
            # This is D[js, :] @ a[i, :]  -> (len(js),)
            T1_new = D[js, :] @ a[i, :]
            
            # Term 2: A[:, i] dot D[:, j] for each j.
            # A[:, i] is (n,). D[:, js] is (n, len(js)).
            # (D[:, js] @ a[:, i]) -> (len(js),)
            T2_new = D[:, js] @ a[:, i]
            
            # Term 3: A[j, :] dot D[i, :] for each j.
            # A[js, :] @ D[i, :] -> (len(js),)
            T3_new = a[js, :] @ D[i, :]
            
            # Term 4: A[:, j] dot D[:, i] for each j.
            # D[:, i] is (n,). A[:, js] is (n, len(js)).
            # (A[:, js] @ D[:, i]) -> (len(js),)
            T4_new = a[:, js] @ D[:, i]
            
            T_new = T1_new + T2_new + T3_new + T4_new
            
            # Now T_old:
            # Term 1: A[i, :] dot D[i, :] (constant for all j)
            T1_old = a[i, :] @ D[i, :]
            
            # Term 2: A[:, i] dot D[:, i] (constant)
            T2_old = a[:, i] @ D[:, i]
            
            # Term 3: A[j, :] dot D[j, :] for each j.
            # This is sum_l A[j,l] D[j,l].
            # (a[js, :] * D[js, :]).sum(axis=1)
            T3_old = np.sum(a[js, :] * D[js, :], axis=1)
            
            # Term 4: A[:, j] dot D[:, j] for each j.
            # (a[:, js] * D[:, js]).sum(axis=0)
            T4_old = np.sum(a[:, js] * D[:, js], axis=0)
            
            T_old = T1_old + T2_old + T3_old + T4_old
            
            # Wait, this double counts the intersections?
            # In T_old, T1 includes A[i,j]D[i,j]. T4 includes A[i,j]D[i,j] (since k=i, l=j -> A[i,j]D[i,j]).
            # So A[i,j]D[i,j] is counted twice.
            # Similarly, T2 includes A[j,i]D[j,i]. T3 includes A[j,i]D[j,i].
            # So A[j,i]D[j,i] is counted twice.
            
            # In T_new:
            # T1 includes A[i,j]D[j,j]? No, T1 is A[i,:] dot D[j,:]. Includes A[i,j]D[j,j].
            # T4 includes A[i,j]D[j,i]? No, T4 is A[:,j] dot D[:,i]. Includes A[i,j]D[i,i]? No.
            # Let's trace T4_new for a specific j.
            # T4_new[j] = sum_k A[k,j] D[k,i]. Includes k=i -> A[i,j]D[i,i].
            # T1_new[j] = sum_l A[i,l] D[j,l]. Includes l=j -> A[i,j]D[j,j].
            
            # This approach is error-prone. Let's stick to the previous "Cross + Diagonal + Direct" method but ensure correctness.
            
            # Correct Delta Formula for swapping positions i and j:
            # Let u=p[i], v=p[j].
            # Delta = 
            #   (A[i,i] - A[j,j]) * (B[v,v] - B[u,u])
            # + (A[i,j] - A[j,i]) * (B[v,u] - B[u,v])
            # + sum_{k!=i,j} [ (A[i,k] - A[j,k]) * (B[v,p[k]] - B[u,p[k]]) 
            #                  + (A[k,i] - A[k,j]) * (B[p[k],v] - B[p[k],u]) ]
            
            # In terms of D (where D[x,y] = B[p[x], p[y]]):
            # B[u,u] = D[i,i], B[v,v] = D[j,j]
            # B[u,v] = D[i,j], B[v,u] = D[j,i]
            # B[u,p[k]] = D[i,k], B[v,p[k]] = D[j,k]
            # B[p[k],u] = D[k,i], B[p[k],v] = D[k,j]
            
            # Delta = 
            #   (A[i,i] - A[j,j]) * (D[j,j] - D[i,i])
            # + (A[i,j] - A[j,i]) * (D[j,i] - D[i,j])
            # + sum_{k!=i,j} [ (A[i,k] - A[j,k]) * (D[j,k] - D[i,k]) 
            #                  + (A[k,i] - A[k,j]) * (D[k,j] - D[k,i]) ]
            
            # Let's compute the sum over ALL k first, then subtract k=i and k=j terms.
            
            # Sum_all_k = sum_k [ (A[i,k] - A[j,k]) * (D[j,k] - D[i,k]) 
            #                  + (A[k,i] - A[k,j]) * (D[k,j] - D[k,i]) ]
            
            # Vectorized for all j in js:
            # Part 1: sum_k (A[i,k] - A[j,k]) * (D[j,k] - D[i,k])
            # A[i,:] - A[js,:] -> (len(js), n)
            # D[js,:] - D[i,:] -> (len(js), n)
            # Sum over k (axis=1)
            P1 = np.sum((a[i, :] - a[js, :]) * (D[js, :] - D[i, :]), axis=1)
            
            # Part 2: sum_k (A[k,i] - A[k,j]) * (D[k,j] - D[k,i])
            # A[:,i] - A[:,js] -> (n, len(js))
            # D[:,js] - D[:,i] -> (n, len(js))
            # Sum over k (axis=0)
            P2 = np.sum((a[:, i] - a[:, js]) * (D[:, js] - D[:, i]), axis=0)
            
            Sum_all = P1 + P2
            
            # Now subtract k=i and k=j terms from Sum_all.
            
            # k=i term in P1: (A[i,i] - A[j,i]) * (D[j,i] - D[i,i])
            # k=i term in P2: (A[i,i] - A[i,j]) * (D[i,j] - D[i,i])
            
            # k=j term in P1: (A[i,j] - A[j,j]) * (D[j,j] - D[i,j])
            # k=j term in P2: (A[j,i] - A[j,j]) * (D[j,j] - D[j,i])
            
            # This is still messy because P1 and P2 are vectors over j.
            
            # Let's just compute the delta for each j individually if vectorization is too complex/error-prone.
            # n=256, so n^2/2 = 32k swaps. Python loop might be too slow.
            # But we can use NumPy to compute the delta for a single i against all j > i.
            
            # Let's try a different vectorization.
            # For fixed i, we want Delta(i, j) for all j > i.
            
            # Let's compute the "Cross" term for k not in {i,j} by summing over all k and then correcting.
            
            # Actually, let's just use the formula:
            # Delta(i, j) = (A[i,i] - A[j,j])*(D[j,j] - D[i,i]) + (A[i,j] - A[j,i])*(D[j,i] - D[i,j]) + Cross(i, j)
            # where Cross(i, j) = sum_{k!=i,j} [ (A[i,k] - A[j,k])*(D[j,k] - D[i,k]) + (A[k,i] - A[k,j])*(D[k,j] - D[k,i]) ]
            
            # We can compute Cross(i, j) for all j > i as follows:
            # Let S1(j) = sum_k (A[i,k] - A[j,k])*(D[j,k] - D[i,k])
            # Let S2(j) = sum_k (A[k,i] - A[k,j])*(D[k,j] - D[k,i])
            
            # S1(j) includes k=i and k=j.
            # S1(i) part: (A[i,i] - A[j,i])*(D[j,i] - D[i,i])
            # S1(j) part: (A[i,j] - A[j,j])*(D[j,j] - D[i,j])
            
            # S2(j) includes k=i and k=j.
            # S2(i) part: (A[i,i] - A[i,j])*(D[i,j] - D[i,i])
            # S2(j) part: (A[j,i] - A[j,j])*(D[j,j] - D[j,i])
            
            # So Cross(i, j) = S1(j) + S2(j) - [S1_i_part + S1_j_part + S2_i_part + S2_j_part]
            
            # Let's compute S1 and S2 for all j > i.
            
            # S1(j) = sum_k (A[i,k] - A[j,k])*(D[j,k] - D[i,k])
            # = sum_k A[i,k]D[j,k] - sum_k A[i,k]D[i,k] - sum_k A[j,k]D[j,k] + sum_k A[j,k]D[i,k]
            # = (D[j,:] @ A[i,:]) - (D[i,:] @ A[i,:]) - (D[j,:] @ A[j,:]) + (D[i,:] @ A[j,:])
            
            # Vectorized for all j in js:
            # D[js,:] @ A[i,:] -> (len(js),)
            # D[i,:] @ A[i,:] -> scalar
            # (D[js,:] * A[js,:]).sum(axis=1) -> (len(js),)
            # D[i,:] @ A[js,:] -> (len(js),)
            
            S1 = (D[js, :] @ a[i, :]) - (D[i, :] @ a[i, :]) - np.sum(D[js, :] * a[js, :], axis=1) + (D[i, :] @ a[js, :].T)
            
            # S2(j) = sum_k (A[k,i] - A[k,j])*(D[k,j] - D[k,i])
            # = sum_k A[k,i]D[k,j] - sum_k A[k,i]D[k,i] - sum_k A[k,j]D[k,j] + sum_k A[k,j]D[k,i]
            # = (D[:,j] @ A[:,i]) - (D[:,i] @ A[:,i]) - (D[:,j] @ A[:,j]) + (D[:,i] @ A[:,j])
            
            # Vectorized for all j in js:
            # D[:,js] @ A[:,i] -> (len(js),)
            # D[:,i] @ A[:,i] -> scalar
            # (D[:,js] * A[:,js]).sum(axis=0) -> (len(js),)
            # D[:,i] @ A[:,js] -> (len(js),)
            
            S2 = (D[:, js] @ a[:, i]) - (D[:, i] @ a[:, i]) - np.sum(D[:, js] * a[:, js], axis=0) + (D[:, i] @ a[:, js])
            
            S_total = S1 + S2
            
            # Now subtract the i and j parts.
            
            # S1_i_part(j) = (A[i,i] - A[j,i]) * (D[j,i] - D[i,i])
            # S1_j_part(j) = (A[i,j] - A[j,j]) * (D[j,j] - D[i,j])
            # S2_i_part(j) = (A[i,i] - A[i,j]) * (D[i,j] - D[i,i])
            # S2_j_part(j) = (A[j,i] - A[j,j]) * (D[j,j] - D[j,i])
            
            # These are vectors of length len(js).
            
            S1_i = (a[i, i] - a[js, i]) * (D[js, i] - D[i, i])
            S1_j = (a[i, js] - np.diag(a[js, :])) * (D[js, js] - D[i, js]) # Wait, A[j,j] is diag of A for row j.
            # A[j,j] for j in js is a[js, js] element-wise.
            S1_j = (a[i, js] - np.diag(a[np.ix_(js, js)])) * (D[np.ix_(js, js)] - D[i, js])
            
            S2_i = (a[i, i] - a[i, js]) * (D[i, js] - D[i, i])
            S2_j = (a[js, i] - np.diag(a[np.ix_(js, js)])) * (D[np.ix_(js, js)] - D[js, i])
            
            Subtractions = S1_i + S1_j + S2_i + S2_j
            
            Cross = S_total - Subtractions
            
            # Direct terms:
            Direct = (a[i, i] - np.diag(a[np.ix_(js, js)])) * (D[np.ix_(js, js)] - D[i, i]) \
                   + (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            Deltas = Cross + Direct
            
            min_idx = np.argmin(Deltas)
            min_delta = Deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
            
        if not improved:
            break
            
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # --- Phase 0: Initialization ---
    # Use spectral embedding for a strong initial guess
    try:
        A_sym_init = a + a.T
        B_sym_init = b + b.T

        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym_init)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym_init)

        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]

        V_A = eigvecs_A[:, idx_A]
        V_B = eigvecs_B[:, idx_B]

        C_spectral = -V_A @ V_B.T
        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
    except Exception:
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 1: Frank-Wolfe ---
    fw_deadline = start_time + 30.0
    is_sym = np.allclose(b, b.T)
    
    while time.time() < fw_deadline:
        B_p = b[p]
        if is_sym:
            A_sym = a + a.T
            C = A_sym @ B_p
        else:
            C = a @ B_p + a.T @ B_p.T

        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind

        new_cost = calc_cost(p_new)
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            break

    # --- Phase 2: Local Search + ILS ---
    p = best_p.copy()
    current_cost = best_cost
    
    # Run local search on the best solution found so far
    if time.time() < deadline - 5.0:
        p_opt, cost_opt = _two_opt_search(a, b, p, time_limit=deadline - 10.0, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()

    # ILS Loop
    rng = np.random.default_rng(42)
    base_perturb = 3
    consecutive_no_improve = 0
    
    while time.time() < deadline - 5.0:
        if consecutive_no_improve > 5:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = best_p.copy()
        for _ in range(perturb_k):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Run local search on perturbed solution
        # Allocate a small time chunk
        ls_time_limit = min(10.0, deadline - 5.0 - (time.time() - start_time))
        if ls_time_limit < 2.0:
            break
            
        p_opt, cost_opt = _two_opt_search(a, b, p_pert, time_limit=ls_time_limit, start_time=time.time())
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
