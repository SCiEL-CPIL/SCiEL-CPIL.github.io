def _local_search(a, b, p, time_limit, start_time, max_passes=100):
    """
    Local search with 2-opt moves for QAP.
    Correct vectorized delta computation.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute A row/col sums? No, we need specific terms.
    # To speed up, we can precompute nothing complex, just use the formula.
    
    passes = 0
    while time.time() - start_time < time_limit and passes < max_passes:
        improved = False
        passes += 1
        
        # Iterate through all pairs (i, j) with i < j
        # We can vectorize the calculation of deltas for a fixed i against all j > i
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
            
            # We want to compute delta for swapping i with each j in js.
            # Formula:
            # Delta_ij = sum_k (A_ik + A_ki - A_jk - A_kj) * (D_jk - D_ik) 
            #          + sum_k (A_ik + A_ki - A_jk - A_kj) * (D_kj - D_ki)
            #          + A_ij * (D_ji + D_ij - D_ii - D_jj)
            #          + A_ji * (D_ij + D_ji - D_jj - D_ii)
            
            # Let V_k = A_ik + A_ki - A_jk - A_kj. This depends on j, so we can't precompute a single V for all j.
            # V(i, j, k) = A[i,k] + A[k,i] - A[j,k] - A[k,j]
            
            # Term 1: sum_k V(i,j,k) * (D[j,k] - D[i,k])
            # Term 2: sum_k V(i,j,k) * (D[k,j] - D[k,i])
            
            # Let's vectorize over j for fixed i.
            # A[i, :] is (n,)
            # A[:, i] is (n,)
            # A[js, :] is (len(js), n)
            # A[:, js] is (n, len(js))
            
            # V matrix shape: (len(js), n)
            # V[j_idx, k] = A[i,k] + A[k,i] - A[js[j_idx], k] - A[k, js[j_idx]]
            
            # A[i,k] + A[k,i] is shape (n,)
            # A[js, :] is (len(js), n)
            # A[:, js].T is (len(js), n)
            
            V = (a[i, :] + a[:, i]) - (a[js, :] + a[:, js].T)
            # V is (len(js), n)
            
            # D_part1: D[js, :] - D[i, :] -> (len(js), n)
            D_diff1 = D[js, :] - D[i, :]
            
            # D_part2: D[:, js] - D[:, i] -> (n, len(js)). We need (len(js), n) for dot with V?
            # Term 2 is sum_k V[j,k] * (D[k,j] - D[k,i])
            # D[:, js] is (n, len(js)). D[:, js].T is (len(js), n).
            # D[:, i] is (n,). D[:, i].T is (n,).
            # We need (D[k,j] - D[k,i]) for each k.
            # D[:, js] - D[:, i] is (n, len(js)).
            # Transpose to (len(js), n).
            D_diff2 = (D[:, js] - D[:, i]).T
            
            # Term 1 sum: (V * D_diff1).sum(axis=1) -> (len(js),)
            t1 = (V * D_diff1).sum(axis=1)
            
            # Term 2 sum: (V * D_diff2).sum(axis=1) -> (len(js),)
            t2 = (V * D_diff2).sum(axis=1)
            
            # Term 3 & 4: Cross terms involving i and j directly.
            # A[i, js] is (len(js),)
            # A[js, i] is (len(js),)
            # D[i, js] is (len(js),)
            # D[js, i] is (len(js),)
            # D[i, i] is scalar
            # D[js, js] is (len(js),)
            
            cross = a[i, js] * (D[js, i] + D[i, js] - D[i, i] - D[js, js])
            cross += a[js, i] * (D[i, js] + D[js, i] - D[js, js] - D[i, i])
            
            deltas = t1 + t2 + cross
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Swap i and j in p
                p[i], p[j] = p[j], p[i]
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost
