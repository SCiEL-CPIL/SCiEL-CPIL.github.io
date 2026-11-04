    def local_search(p, cost, max_passes=50):
        """
        Vectorized Best-Improvement 2-opt Local Search.
        """
        p = p.copy()
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # Precompute matrices for fast delta calculation
            # M_loc[x, k] = B[x, p[k]]
            M_loc = B[:, p]
            # H[i, x] = sum_k A[i, k] M_loc[x, k] = sum_k A[i, k] B[x, p[k]]
            H = A @ M_loc.T
            
            # K[k, y] = B[p[k], y]
            K = B[p, :]
            # G[i, y] = sum_k A[k, i] K[k, y] = sum_k A[k, i] B[p[k], y]
            G = A.T @ K
            
            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost
                
                # Calculate delta for swapping i with all j > i
                # We only care about j > i to avoid duplicates and self-swap
                js = np.arange(i + 1, n)
                
                # Vectorized calculation of SumPart1 and SumPart2
                # H[i, p_j] -> H[i, p[js]]
                # H[i, p_i] -> scalar
                # H[j, p_j] -> H[js, p[js]]
                # H[j, p_i] -> H[js, p_i]
                
                h_ipj = H[i, p[js]]
                h_ipi = H[i, p[i]]
                h_jpj = H[js, p[js]]
                h_jpi = H[js, p[i]]
                
                sum1 = h_ipj - h_ipi - h_jpj + h_jpi
                
                g_ipj = G[i, p[js]]
                g_ipi = G[i, p[i]]
                g_jpj = G[js, p[js]]
                g_jpi = G[js, p[i]]
                
                sum2 = g_ipj - g_ipi - g_jpj + g_jpi
                
                # Corrections
                # corr1_i: (A[i, i] - A[j, i]) * (B[p_j, p_i] - B[p_i, p_i])
                # Note: A[j, i] varies with j.
                # B[p_j, p_i] varies with j.
                
                a_ii = A[i, i]
                a_ji = A[js, i] # Vector
                b_pj_pi = B[p[js], p[i]] # Vector
                b_pi_pi = B[p[i], p[i]] # Scalar
                
                corr1_i = (a_ii - a_ji) * (b_pj_pi - b_pi_pi)
                
                # corr1_j: (A[i, j] - A[j, j]) * (B[p_j, p_j] - B[p_i, p_j])
                a_ij = A[i, js] # Vector
                a_jj = A[js, js] # Vector
                b_pj_pj = B[p[js], p[js]] # Vector
                b_pi_pj = B[p[i], p[js]] # Vector
                
                corr1_j = (a_ij - a_jj) * (b_pj_pj - b_pi_pj)
                
                # corr2_i: (A[i, i] - A[i, j]) * (B[p_i, p_j] - B[p_i, p_i])
                b_pi_pj2 = B[p[i], p[js]] # Same as b_pi_pj
                corr2_i = (a_ii - a_ij) * (b_pi_pj2 - b_pi_pi)
                
                # corr2_j: (A[j, i] - A[j, j]) * (B[p_j, p_j] - B[p_j, p_i])
                a_ji2 = A[js, i] # Same as a_ji
                b_pj_pi2 = B[p[js], p[i]] # Same as b_pj_pi
                corr2_j = (a_ji2 - a_jj) * (b_pj_pj - b_pj_pi2)
                
                # Diag and Cross
                # diag: (A[i, i] - A[j, j]) * (B[p_j, p_j] - B[p_i, p_i])
                diag = (a_ii - a_jj) * (b_pj_pj - b_pi_pi)
                
                # cross: (A[i, j] - A[j, i]) * (B[p_j, p_i] - B[p_i, p_j])
                cross = (a_ij - a_ji) * (b_pj_pi - b_pi_pj)
                
                deltas = sum1 + sum2 - corr1_i - corr1_j - corr2_i - corr2_j + diag + cross
                
                # Find best j
                min_idx = np.argmin(deltas)
                best_delta = deltas[min_idx]
                
                if best_delta < 0:
                    best_j = js[min_idx]
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += int(best_delta)
                    improved = True
                    break # Restart pass for best improvement
                    
        return p, cost
