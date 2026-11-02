<<<<<<< SEARCH
    U36, V36, W36 = get_solution_rank36()
    if verify_solution(U36, V36, W36):
        U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
        if verify_solution(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # Fallback: schoolbook
    rank = nm * mp
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    U36, V36, W36 = get_solution_rank36()
    if verify_solution(U36, V36, W36):
        if best_U is None:
            U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
            if verify_solution(U_snap, V_snap, W_snap):
                best_U, best_V, best_W = U_snap, V_snap, W_snap
            else:
                best_U, best_V, best_W = U36, V36, W36
    
    # Time-budgeted search for rank reduction via linear dependency detection
    if time.time() - start_time < time_limit - 15:
        if best_U is not None:
            R = best_U.shape[1]
            # Build the 3D tensor T[a,b,c,d,e,f] = sum_mu U[a,b,mu]*V[c,d,mu]*W[e,f,mu]
            # Reshape: U is (nm, R) -> (n,m,R), V is (mp,R) -> (m,p,R), W is (pn,R) -> (p,n,R)
            U3 = best_U.reshape(n, m, R)
            V3 = best_V.reshape(m, p, R)
            W3 = best_W.reshape(p, n, R)
            
            # The multiplication tensor T_{i,j,k} is a 3-tensor of shape (n*m, m*p, p*n)
            # Flatten to matrix form for SVD-based rank analysis
            # T as a matrix: (n*m*p) x (n*m*p) won't work directly
            # Instead, use the slice approach: for each fixed k, T[:, :, k] should be the identity pattern
            
            # Better approach: check if columns of the factor matrices are linearly dependent
            # by checking if removing any single column still gives a valid decomposition
            
            # For each column mu, check if it can be expressed as a linear combination of others
            # by checking if the "residual" tensor (with that column removed) is still valid
            # This is expensive, so use a smarter approach:
            
            # Check rank of the 3-way tensor via unfolding
            # Unfold along mode 0: matrix of shape (n*m) x (m*p * p*n)
            T_unfold0 = np.zeros((n*m, m*p*p*n))
            T_unfold1 = np.zeros((m*p, n*m*p*n))
            T_unfold2 = np.zeros((p*n, n*m*m*p))
            
            for mu in range(R):
                # u_mu: (n*m,), v_mu: (m*p,), w_mu: (p*n,)
                u_mu = best_U[:, mu]
                v_mu = best_V[:, mu]
                w_mu = best_W[:, mu]
                
                # T_unfold0[:, j*pn+k] for each (j,k) pair... this is complex
                # Use einsum for efficiency
                pass
            
            # Simpler: just try removing pairs of columns and check validity
            # Only do this if we have time
            if time.time() - start_time < time_limit - 20:
                # Try removing each single column
                for mu in range(R):
                    if time.time() - start_time > time_limit - 20:
                        break
                    U_test = np.delete(best_U, mu, axis=1)
                    V_test = np.delete(best_V, mu, axis=1)
                    W_test = np.delete(best_W, mu, axis=1)
                    if verify_solution(U_test, V_test, W_test):
                        best_U, best_V, best_W = U_test, V_test, W_test
                        R = best_U.shape[1]
                        # Restart search with reduced rank
                        break
                
                # Try removing pairs of columns (only if time permits)
                if time.time() - start_time < time_limit - 30:
                    for mu1 in range(best_U.shape[1]):
                        if time.time() - start_time > time_limit - 25:
                            break
                        for mu2 in range(mu1 + 1, best_U.shape[1]):
                            if time.time() - start_time > time_limit - 25:
                                break
                            idx = [i for i in range(best_U.shape[1]) if i != mu1 and i != mu2]
                            U_test = best_U[:, idx]
                            V_test = best_V[:, idx]
                            W_test = best_W[:, idx]
                            if verify_solution(U_test, V_test, W_test):
                                best_U, best_V, best_W = U_test, V_test, W_test
                                break
                        if best_U.shape[1] < R:
                            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    rank = nm * mp
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

