import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n

    # Helper to check if a decomposition is valid
    def check_validity_residual(U, V, W, tol=1e-8):
        R = U.shape[1]
        V_mat = V.reshape(m, p, R)  # V_mat[j, k, r]
        U_mat = U.reshape(n, m, R)  # U_mat[i, j, r]
        W_mat = W.reshape(p, n, R)  # W_mat[k, i, r]
        
        err = 0.0
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Naive part: A_right (2x1) * B_bottom (1x10) -> 2*10 = 20 multiplications
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2  # A[i, 2]
                v_idx = 2 * p + k  # B[2, k]
                w_idx = k * n + i  # C[k, i]
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
        
        # Strassen part: A_left (2x2) * B_top (2x10)
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # M1 = (a11+a22)(b11+b22)
            U[0*m+0, col] += 1
            U[1*m+1, col] += 1
            V[0*p+k0, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            # M2 = (a21+a22)b11
            U[1*m+0, col] += 1
            U[1*m+1, col] += 1
            V[0*p+k0, col] += 1
            W[k0*n+1, col] += 1
            W[k1*n+1, col] += -1
            col += 1
            
            # M3 = a11(b12-b22)
            U[0*m+0, col] += 1
            V[0*p+k1, col] += 1
            V[1*p+k1, col] += -1
            W[k1*n+0, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            # M4 = a22(b21-b11)
            U[1*m+1, col] += 1
            V[1*p+k0, col] += 1
            V[0*p+k0, col] += -1
            W[k0*n+0, col] += 1
            W[k0*n+1, col] += 1
            col += 1
            
            # M5 = (a11+a12)b22
            U[0*m+0, col] += 1
            U[0*m+1, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += -1
            W[k1*n+0, col] += 1
            col += 1
            
            # M6 = (a21-a11)(b11+b12)
            U[1*m+0, col] += 1
            U[0*m+0, col] += -1
            V[0*p+k0, col] += 1
            V[0*p+k1, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            # M7 = (a12-a22)(b21+b22)
            U[0*m+1, col] += 1
            U[1*m+1, col] += -1
            V[1*p+k0, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += 1
            col += 1
        
        return U, V, W

    # Get the known rank 55 solution
    U_55, V_55, W_55 = get_rank_55_solution()
    err_55 = check_validity_residual(U_55, V_55, W_55)
    
    if err_55 > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U_55, V_55, W_55
    best_rank = 55

    # Attempt to find lower rank solutions using ALS
    def als_lower_rank(target_rank, max_iters=20, tol=1e-10):
        if time.time() > deadline - 10:
            return None
        
        # Initialize randomly
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                # M[j*p+k, c] = V[j*p+k, c] * W[k*n+i, c]
                M_mat = np.zeros((m*p, target_rank))
                for j in range(m):
                    for k in range(p):
                        M_mat[j*p+k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve M_mat @ x = ones(m*p)
                try:
                    sol = np.linalg.lstsq(M_mat, np.ones(m*p), rcond=None)[0]
                    U_new[i*m:(i+1)*m, :] = sol
                except:
                    return None
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # N[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    N_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        N_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(N_mat, np.ones(n), rcond=None)[0]
                        V_new[j*p+k, :] = sol
                    except:
                        return None
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # O[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    O_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        O_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol = np.linalg.lstsq(O_mat, np.ones(m), rcond=None)[0]
                        W_new[k*n+i, :] = sol
                    except:
                        return None
            W = W_new
            
            # Check convergence
            err = check_validity_residual(U, V, W)
            if err < tol:
                return U, V, W
                
        return None

    # Try ranks from 54 down to 50
    for target_rank in range(54, 49, -1):
        if time.time() > deadline - 15:
            break
            
        # Try multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            result = als_lower_rank(target_rank)
            if result is not None:
                U_new, V_new, W_new = result
                err = check_validity_residual(U_new, V_new, W_new)
                if err < 1e-8:
                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    # Return the rank-55 solution
    return {"U": U_55.tolist(), "V": V_55.tolist(), "W": W_55.tolist()}
