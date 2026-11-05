<<<<<<< SEARCH
    # Build full decomposition with rank 36
    U1_s1, V1_s1, W1_s1 = get_strassen_222(0, 0)
    U1_s2, V1_s2, W1_s2 = get_strassen_222(0, 2)
    U1_t1, V1_t1, W1_t1 = get_trivial_221(0, 4)
    
    U_block1 = np.hstack([U1_s1, U1_s2, U1_t1])
    V_block1 = np.hstack([V1_s1, V1_s2, V1_t1])
    W_block1 = np.hstack([W1_s1, W1_s2, W1_t1])

    U2_s1, V2_s1, W2_s1 = get_strassen_222(2, 0)
    U2_s2, V2_s2, W2_s2 = get_strassen_222(2, 2)
    U2_t1, V2_t1, W2_t1 = get_trivial_221(2, 4)
    
    U_block2 = np.hstack([U2_s1, U2_s2, U2_t1])
    V_block2 = np.hstack([V2_s1, V2_s2, V2_t1])
    W_block2 = np.hstack([W2_s1, W2_s2, W2_t1])

    U_full = np.hstack([U_block1, U_block2])
    V_full = np.hstack([V_block1, V_block2])
    W_full = np.hstack([W_block1, W_block2])

    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    U_s, V_s, W_s = schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
=======
    # Strategy 1: Block decomposition with Strassen (rank 36)
    def build_strassen_blocks():
        U1_s1, V1_s1, W1_s1 = strassen_222(0, 0)
        U1_s2, V1_s2, W1_s2 = strassen_222(0, 2)
        U1_t1, V1_t1, W1_t1 = trivial_221(0, 4)
        
        U_block1 = np.hstack([U1_s1, U1_s2, U1_t1])
        V_block1 = np.hstack([V1_s1, V1_s2, V1_t1])
        W_block1 = np.hstack([W1_s1, W1_s2, W1_t1])

        U2_s1, V2_s1, W2_s1 = strassen_222(2, 0)
        U2_s2, V2_s2, W2_s2 = strassen_222(2, 2)
        U2_t1, V2_t1, W2_t1 = trivial_221(2, 4)
        
        U_block2 = np.hstack([U2_s1, U2_s2, U2_t1])
        V_block2 = np.hstack([V2_s1, V2_s2, V2_t1])
        W_block2 = np.hstack([W2_s1, W2_s2, W2_t1])

        U_full = np.hstack([U_block1, U_block2])
        V_full = np.hstack([V_block1, V_block2])
        W_full = np.hstack([W_block1, W_block2])
        return U_full, V_full, W_full

    U_full, V_full, W_full = build_strassen_blocks()
    if verify(U_full, V_full, W_full):
        best_rank = U_full.shape[1]
    else:
        U_s, V_s, W_s = schoolbook()
        best_rank = U_s.shape[1]
        U_full, V_full, W_full = U_s, V_s, W_s

    # Try to find a lower rank decomposition via random search
    # The slice-rank lower bound is max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Try ranks from 20 up to best_rank-1
    if time.time() < deadline - 30:
        np.random.seed(42)
        for target_rank in range(20, best_rank):
            if time.time() > deadline - 30:
                break
            # Try a few random initializations
            for attempt in range(50):
                if time.time() > deadline - 30:
                    break
                U_r = np.random.randn(n * m, target_rank)
                V_r = np.random.randn(m * p, target_rank)
                W_r = np.random.randn(p * n, target_rank)
                
                # Use gradient-free optimization: iteratively update one factor at a time
                # For fixed V, W: U[i*m+j, :] should satisfy
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is: U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) = 1 for all k
                # So we need U[i*m+j, :] to be a solution to a linear system.
                
                converged = False
                for it in range(200):
                    if time.time() > deadline - 30:
                        break
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            # Solve for U[i*m+j, :] such that
                            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                            A_mat = np.zeros((p, target_rank))
                            for k in range(p):
                                A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            # Least squares
                            try:
                                U_r[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                            except:
                                pass
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((n, target_rank))
                            for i in range(n):
                                A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            try:
                                V_r[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                            except:
                                pass
                    
                    # Update W
                    for k in range(p):
                        for i in range(n):
                            A_mat = np.zeros((m, target_rank))
                            for j in range(m):
                                A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            try:
                                W_r[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                            except:
                                pass
                    
                    # Check convergence
                    if it % 20 == 19:
                        err = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                                    err += abs(val - 1.0)
                        if err < 1e-6:
                            converged = True
                            break
                
                if converged and verify(U_r, V_r, W_r):
                    best_rank = target_rank
                    U_full, V_full, W_full = U_r.copy(), V_r.copy(), W_r.copy()
                    break
            if best_rank == target_rank:
                break

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
>>>>>>> REPLACE
