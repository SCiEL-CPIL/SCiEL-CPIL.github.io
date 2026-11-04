def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = 120
    
    n, m, p = 2, 4, 6
    nm, mp, pn = n*m, m*p, p*n  # 8, 24, 12
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        if time.time() > deadline - 5:
            return False
        # Check: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(rank):
                        total += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    def add_strassen(i0, j0, k0, off, U, V, W):
        a00, a01 = i0*m+j0, i0*m+j0+1
        a10, a11 = (i0+1)*m+j0, (i0+1)*m+j0+1
        b00, b01 = j0*p+k0, j0*p+k0+1
        b10, b11 = (j0+1)*p+k0, (j0+1)*p+k0+1
        w00, w01 = k0*n+i0, (k0+1)*n+i0
        w10, w11 = k0*n+i0+1, (k0+1)*n+i0+1
        
        U[a00,off]+=1;U[a11,off]+=1;V[b00,off]+=1;V[b11,off]+=1;W[w00,off]+=1;W[w11,off]+=1
        U[a10,off+1]+=1;U[a11,off+1]+=1;V[b00,off+1]+=1;W[w10,off+1]+=1;W[w11,off+1]-=1
        U[a00,off+2]+=1;V[b01,off+2]+=1;V[b11,off+2]-=1;W[w01,off+2]+=1;W[w11,off+2]+=1
        U[a11,off+3]+=1;V[b10,off+3]+=1;V[b00,off+3]-=1;W[w00,off+3]+=1;W[w10,off+3]+=1
        U[a00,off+4]+=1;U[a01,off+4]+=1;V[b11,off+4]+=1;W[w00,off+4]-=1;W[w01,off+4]+=1
        U[a10,off+5]+=1;U[a00,off+5]-=1;V[b00,off+5]+=1;V[b01,off+5]+=1;W[w11,off+5]+=1
        U[a01,off+6]+=1;U[a11,off+6]-=1;V[b10,off+6]+=1;V[b11,off+6]+=1;W[w00,off+6]+=1
    
    # First try: rank 42 Strassen blocks (known good)
    U = np.zeros((nm, 42))
    V = np.zeros((mp, 42))
    W = np.zeros((pn, 42))
    off = 0
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                add_strassen(i0, j0, k0, off, U, V, W)
                off += 7
    
    if verify(U, V, W, 42):
        # Try to reduce rank by solving linear systems
        # Strategy: try rank 40, 39, 38, ...
        best_rank = 42
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for target_rank in range(41, 35, -1):
            if time.time() > deadline - 15:
                break
            success = False
            
            # Try multiple random initializations
            for attempt in range(20):
                if time.time() > deadline - 15:
                    break
                
                # Start from rank-42, zero out random columns
                U_trunc = U[:, :target_rank].copy()
                V_trunc = V[:, :target_rank].copy()
                W_trunc = W[:, :target_rank].copy()
                
                # Random perturbation to escape local minima
                if attempt > 0:
                    U_trunc += np.random.randn(nm, target_rank) * 0.1
                    V_trunc += np.random.randn(mp, target_rank) * 0.1
                    W_trunc += np.random.randn(pn, target_rank) * 0.1
                
                # Alternating least squares
                for iteration in range(50):
                    if time.time() > deadline - 15:
                        break
                    
                    # Fix V and W, solve for U
                    # For each (i,j): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all k
                    # This is: U[i*m+j, :] @ (V[j*p:,:]*W[:,i:,:])^T... 
                    # More precisely: for fixed (i,j), define M_{k,t} = V[j*p+k, t] * W[k*n+i, t]
                    # Then: sum_t U[i*m+j, t] * M_{k,t} = 1 for k=0..p-1
                    # This is: U[i*m+j, :] @ M^T = 1^T, i.e., M @ U[i*m+j, :]^T = 1
                    for i in range(n):
                        for j in range(m):
                            # Build M: p x target_rank
                            M = np.zeros((p, target_rank))
                            for k in range(p):
                                M[k, :] = V_trunc[j*p+k, :] * W_trunc[k*n+i, :]
                            # Solve M @ u^T = 1
                            # Least squares solution
                            try:
                                u_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                                U_trunc[i*m+j, :] = u_sol
                            except:
                                pass
                    
                    # Fix U and W, solve for V
                    # For each (j,k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i
                    # For fixed (j,k): sum_t V[j*p+k, t] * (U[i*m+j, t] * W[k*n+i, t]) = 1 for i=0..n-1
                    # Define N_{i,t} = U[i*m+j, t] * W[k*n+i, t]
                    # Then: N @ V[j*p+k, :]^T = 1
                    for j in range(m):
                        for k in range(p):
                            N = np.zeros((n, target_rank))
                            for i in range(n):
                                N[i, :] = U_trunc[i*m+j, :] * W_trunc[k*n+i, :]
                            try:
                                v_sol, residuals, rank_n, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                                V_trunc[j*p+k, :] = v_sol
                            except:
                                pass
                    
                    # Fix U and V, solve for W
                    # For each (k,i): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all j
                    # For fixed (k,i): sum_t W[k*n+i, t] * (U[i*m+j, t] * V[j*p+k, t]) = 1 for j=0..m-1
                    # Define P_{j,t} = U[i*m+j, t] * V[j*p+k, t]
                    # Then: P @ W[k*n+i, :]^T = 1
                    for k in range(p):
                        for i in range(n):
                            P = np.zeros((m, target_rank))
                            for j in range(m):
                                P[j, :] = U_trunc[i*m+j, :] * V_trunc[j*p+k, :]
                            try:
                                w_sol, residuals, rank_p, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                                W_trunc[k*n+i, :] = w_sol
                            except:
                                pass
                    
                    # Check convergence
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for t in range(target_rank):
                                    total += U_trunc[i*m+j, t] * V_trunc[j*p+k, t] * W_trunc[k*n+i, t]
                                max_err = max(max_err, abs(total - 1.0))
                    
                    if max_err < 1e-10:
                        success = True
                        break
                    if max_err < 1e-8:
                        # Close enough, try more iterations
                        continue
                
                if success:
                    best_rank = target_rank
                    best_U = U_trunc.copy()
                    best_V = V_trunc.copy()
                    best_W = W_trunc.copy()
                    break
            
            if success:
                break
        
        if best_rank < 42:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        # Fallback: return rank 42
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: should not reach here
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
