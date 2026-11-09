import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Search for lower rank
    for r in range(rank_school - 1, max(60, rank_school - 25), -1):
        if time.time() > deadline:
            break
            
        for trial in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(1000 + r + trial)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        M_ij = np.zeros((r, p))
                        for k in range(p):
                            M_ij[:, k] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U_new[i*m+j, :] = T[i,j,:] @ np.linalg.pinv(M_ij)
                        except:
                            pass
                U = U_new
                
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        M_jk = np.zeros((r, n))
                        for i in range(n):
                            M_jk[:, i] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V_new[j*p+k, :] = T[:,j,k] @ np.linalg.pinv(M_jk)
                        except:
                            pass
                V = V_new
                
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        M_ki = np.zeros((r, m))
                        for j in range(m):
                            M_ki[:, j] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W_new[k*n+i, :] = T[i,:,k] @ np.linalg.pinv(M_ki)
                        except:
                            pass
                W = W_new
                
            # Check if solution is close to integers/half-integers
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            residual = np.sum((T - recon)**2)
            
            if residual < 1e-4:
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                U_int_reshaped = U_int.reshape(n, m, r)
                V_int_reshaped = V_int.reshape(m, p, r)
                W_int_reshaped = W_int.reshape(p, n, r)
                recon_int = np.einsum('ijr,jkr,kir->ijk', U_int_reshaped, V_int_reshaped, W_int_reshaped)
                exact_residual = np.sum((T - recon_int)**2)
                
                if exact_residual < 1e-9:
                    best_rank = r
                    best_U, best_V, best_W = U_int, V_int, W_int
                    break
                    
            if best_rank < rank_school:
                break
                
        if best_rank < rank_school:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
