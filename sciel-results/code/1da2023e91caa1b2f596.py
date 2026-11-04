def solve_3x3_fast(rank, deadline):
    n = 3
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for seed in range(20):
        if time.time() > deadline - 5: break
        np.random.seed(seed)
        U = np.random.randn(9, rank) * 0.1
        V = np.random.randn(9, rank) * 0.1
        W = np.random.randn(9, rank) * 0.1
        
        for it in range(500):
            if time.time() > deadline - 2: break
            
            # Update U
            # For each i,j, M is 3xRank
            # M[k, r] = V[j*3+k, r] * W[k*3+i, r]
            # We can compute M for all i,j at once?
            # V is 9xR, W is 9xR.
            # V[j*3+k, r] is V.reshape(3,3,R)[j,k,r]
            # W[k*3+i, r] is W.reshape(3,3,R)[k,i,r]
            # M[i,j,k,r] = V3[j,k,r] * W3[k,i,r]
            # We need to solve for U[i,j, r] such that sum_k M[i,j,k,r] U[i,j,r] = 1
            # This is U[i,j, :] = lstsq(M[i,j, :, :].T, ones(3))
            
            V3 = V.reshape(3, 3, rank)
            W3 = W.reshape(3, 3, rank)
            
            # M[i,j,k,r]
            # Loop over i,j is 9 iterations. Rank is 23.
            for i in range(3):
                for j in range(3):
                    # M[k, r]
                    M = np.einsum('kr,kr->kr', V3[j, :, :], W3[:, i, :]) 
                    # Wait, V3[j,k,r] * W3[k,i,r]. Sum over r? No, element-wise product.
                    # M[k, r] = V3[j, k, r] * W3[k, i, r]
                    M = V3[j, :, :] * W3[:, i, :] # Shape (3, Rank)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(3), rcond=None)
                        U[i*3+j, :] = sol
                    except: pass
            
            # Update V
            U3 = U.reshape(3, 3, rank)
            W3 = W.reshape(3, 3, rank)
            for j in range(3):
                for k in range(3):
                    # M[i, r] = U3[i,j,r] * W3[k,i,r]
                    M = U3[:, j, :] * W3[k, :, :] # Shape (3, Rank)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(3), rcond=None)
                        V[j*3+k, :] = sol
                    except: pass
            
            # Update W
            U3 = U.reshape(3, 3, rank)
            V3 = V.reshape(3, 3, rank)
            for k in range(3):
                for i in range(3):
                    # M[j, r] = U3[i,j,r] * V3[j,k,r]
                    M = U3[i, :, :] * V3[:, k, :] # Shape (3, Rank)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(3), rcond=None)
                        W[k*3+i, :] = sol
                    except: pass
            
            # Check error
            if it % 20 == 0:
                # Vectorized error check
                # Sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # U3[i,j,r], V3[j,k,r], W3[k,i,r]
                # T[i,j,k] = sum_r U3[i,j,r] V3[j,k,r] W3[k,i,r]
                # This is a contraction.
                # T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                err = np.sum((T - 1.0)**2)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-8:
                    return U, V, W, err
                    
    return best_U, best_V, best_W, best_err
