def _fast_2opt_polish(a, b, p, time_limit):
    n = len(a)
    p = p.copy()
    start_time = time.time()
    cost = _qap_cost(a, b, p)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                return p, cost
            li = p[i]
            for j in range(i + 1, n):
                lj = p[j]
                
                # Vectorized delta calculation
                # diff_row = b[lj] - b[li]
                # diff_col = b[:, lj] - b[:, li]
                diff_row = b[lj] - b[li]
                diff_col = b[:, lj] - b[:, li]
                
                # We need sum over k != i, j of:
                # (a[i, k] - a[j, k]) * diff_row[p[k]]
                # (a[k, i] - a[k, j]) * diff_col[p[k]]
                
                # Create arrays for the terms
                # term1_k = (a[i, k] - a[j, k]) * diff_row[p[k]]
                # term2_k = (a[k, i] - a[k, j]) * diff_col[p[k]]
                
                a_i_minus_j = a[i] - a[j]
                a_ki_minus_kj = a[:, i] - a[:, j]
                
                # Map diff_row and diff_col to the current permutation p
                # diff_row_mapped[k] = diff_row[p[k]]
                # diff_col_mapped[k] = diff_col[p[k]]
                
                dr_mapped = diff_row[p]
                dc_mapped = diff_col[p]
                
                # Sum over all k
                sum1 = np.dot(a_i_minus_j, dr_mapped)
                sum2 = np.dot(a_ki_minus_kj, dc_mapped)
                
                # Subtract k=i and k=j contributions
                # For k=i:
                # term1_i = (a[i, i] - a[j, i]) * diff_row[p[i]] = (a[i, i] - a[j, i]) * diff_row[li]
                # term2_i = (a[i, i] - a[i, j]) * diff_col[p[i]] = (a[i, i] - a[i, j]) * diff_col[li]
                
                # For k=j:
                # term1_j = (a[i, j] - a[j, j]) * diff_row[p[j]] = (a[i, j] - a[j, j]) * diff_row[lj]
                # term2_j = (a[j, i] - a[j, j]) * diff_col[p[j]] = (a[j, i] - a[j, j]) * diff_col[lj]
                
                sum1 -= (a[i, i] - a[j, i]) * diff_row[li]
                sum1 -= (a[i, j] - a[j, j]) * diff_row[lj]
                
                sum2 -= (a[i, i] - a[i, j]) * diff_col[li]
                sum2 -= (a[j, i] - a[j, j]) * diff_col[lj]
                
                cross_sum = sum1 + sum2
                
                # Self terms
                self_delta = a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
                
                delta = cross_sum + self_delta
                
                if delta < -1e-9:
                    # Swap
                    p[i], p[j] = lj, li
                    cost += delta
                    improved = True
        
        if not improved:
            break
            
    return p, cost
