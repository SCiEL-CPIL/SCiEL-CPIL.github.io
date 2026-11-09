import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n*m*p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ALS for lower ranks
    for R in range(24, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 100 + trial)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i] for all k
                        # This is: U[i*m+j, :] * (V[j*p+0:m*p, :] * W[0:pn, :] elementwise) ... 
                        # More precisely: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # Let A_k = V[j*p+k, :] * W[k*n+i, :] (elementwise product, shape R)
                        # Then: U[i*m+j, :] @ A_k = 1 for all k
                        # This is a linear system: (A_0, A_1, ..., A_{p-1})^T @ U[i*m+j, :]^T = ones(p)
                        # i.e., M @ x = b where M is p×R, x is R, b is ones(p)
                        M_mat = np.zeros((p, R))
                        for k in range(p):
                            M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        # Solve M_mat @ x = b in least squares sense
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        U[i*m+j, :] = x
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        M_mat = np.zeros((n, R))
                        for i in range(n):
                            M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        V[j*p+k, :] = x
                
                # Update W: for each (i,k), solve for W[k*n+i, :]
                for i in range(n):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                        M_mat = np.zeros((m, R))
                        for j in range(m):
                            M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        W[k*n+i, :] = x
                
                # Check convergence
                # Compute residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            residual += abs(val - 1.0)
                if residual < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    # Also check that off-diagonal entries are 0
                    # Actually, we need to check ALL entries of the tensor
                    valid = True
                    for ii in range(nm):
                        for jj in range(mp):
                            for kk in range(pn):
                                val = 0
                                for r in range(R):
                                    val += U_r[ii, r] * V_r[jj, r] * W_r[kk, r]
                                # Check if this should be 1 or 0
                                # T[ii,jj,kk] = 1 if ii=i*m+j, jj=j*p+k, kk=k*n+i for some i,j,k
                                expected = 0.0
                                for i in range(n):
                                    for j in range(m):
                                        for k in range(p):
                                            if ii == i*m+j and jj == j*p+k and kk == k*n+i:
                                                expected = 1.0
                                if abs(val - expected) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
