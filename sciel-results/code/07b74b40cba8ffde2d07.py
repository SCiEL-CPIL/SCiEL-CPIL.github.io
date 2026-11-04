<<<<<<< SEARCH
def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Local search with vectorized delta computation.
    Uses first-improvement for speed, alternating with best-improvement.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    a_sum = a + a.T
    diag_a = np.diag(a).copy()
    
    # Precompute diag_D and update incrementally
    diag_D = np.diag(D).copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            # Compute deltas
            ai_dot_Di = np.dot(a_sum[i], D[i])
            ai_dot_D_all = a_sum[i] @ D
            aj_dot_Dj = np.sum(a_sum * D, axis=1)
            aj_dot_Di = a_sum @ D[i]
            
            S_all = ai_dot_D_all - ai_dot_Di - aj_dot_Dj + aj_dot_Di
            
            coeff_i_all = a_sum[i, i] - a_sum[:, i]
            term_i_all = coeff_i_all * (D[:, i] - D[i, i])
            
            coeff_j_all = a_sum[i, :] - np.diag(a_sum)
            term_j_all = coeff_j_all * (diag_D - D[i, :])
            
            diag_term_all = (a[i, i] - diag_a) * (diag_D - D[i, i])
            
            cross_term_all = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
            
            deltas = S_all - term_i_all - term_j_all + diag_term_all + cross_term_all
            deltas[i] = 0
            
            # Use first-improvement for speed
            best_j = np.argmin(deltas)
            if deltas[best_j] < 0:
                j = best_j
                # Perform swap
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                diag_D[i], diag_D[j] = D[j, j], D[i, i]
                current_cost += deltas[j]
                improved = True
    
    return p, current_cost
=======
def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Local search with vectorized delta computation.
    Uses best-improvement with proper swap updates.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    a_sum = a + a.T
    diag_a = np.diag(a).copy()
    
    # Precompute diag_D and update incrementally
    diag_D = np.diag(D).copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            # Compute deltas for swapping position i with all j
            # delta(i,j) = sum_k [a_sum[i,k]*D[j,k] + a_sum[j,k]*D[i,k]] 
            #             - sum_k [a_sum[i,k]*D[i,k] + a_sum[j,k]*D[j,k]]
            #             - correction terms for k=i and k=j
            
            # Vectorized: for each j, compute the change
            # Main term: (a_sum[i] - a_sum[j]) @ (D[j] - D[i])
            # But we need to subtract terms where k=i or k=j
            
            ai_dot_Di = float(np.dot(a_sum[i], D[i]))
            aj_dot_Dj = float(np.sum(a_sum * D, axis=1)[j] if False else np.dot(a_sum[j], D[j]))
            
            # Vectorized over j
            ai_dot_D_all = a_sum[i] @ D  # shape (n,)
            aj_dot_Di = a_sum @ D[i]     # shape (n,)
            aj_dot_Dj_all = np.sum(a_sum * D, axis=1)  # shape (n,)
            
            # S_all[j] = ai_dot_D_all[j] - ai_dot_Di - aj_dot_Dj_all[j] + aj_dot_Di[j]
            S_all = ai_dot_D_all - ai_dot_Di - aj_dot_Dj_all + aj_dot_Di
            
            # Correction for k=i: (a_sum[i,i] - a_sum[j,i]) * (D[j,i] - D[i,i])
            coeff_i_all = a_sum[i, i] - a_sum[:, i]  # shape (n,)
            term_i_all = coeff_i_all * (D[:, i] - D[i, i])  # shape (n,)
            
            # Correction for k=j: (a_sum[i,j] - a_sum[j,j]) * (D[j,j] - D[i,j])
            coeff_j_all = a_sum[i, :] - a_sum[np.arange(n), np.arange(n)]  # a_sum[i,j] - a_sum[j,j]
            term_j_all = coeff_j_all * (diag_D - D[i, :])  # D[j,j] - D[i,j]
            
            # Diagonal correction: (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            diag_term_all = (a[i, i] - diag_a) * (diag_D - D[i, i])
            
            # Cross term: (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            cross_term_all = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
            
            deltas = S_all - term_i_all - term_j_all + diag_term_all + cross_term_all
            deltas[i] = 0
            
            # Use best-improvement
            best_j = int(np.argmin(deltas))
            if deltas[best_j] < -1e-9:
                j = best_j
                # Perform swap: update p, D, diag_D
                p[i], p[j] = p[j], p[i]
                
                # Update D: swap rows i,j and columns i,j
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                # Update diag_D
                diag_D[i] = D[i, i]
                diag_D[j] = D[j, j]
                
                current_cost += float(deltas[j])
                improved = True
    
    return p, current_cost
>>>>>>> REPLACE

