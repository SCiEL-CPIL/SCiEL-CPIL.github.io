import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 60
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    T = np.ones((n, m, p))
    
    def compute_recon_fast(U, V, W):
        r = U.shape[1]
        recon = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    recon[i,j,:] += u[i,j] * v[j,:] * w[:,i]
        return recon
    
    def err_fast(U, V, W):
        recon = compute_recon_fast(U, V, W)
        return np.max(np.abs(recon - T))
    
    def round_to_half(X):
        return np.round(X * 2) / 2
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    deadline = time.time() + 108
    
    for target_rank in [55, 52, 50, 48]:
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(2):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 7919 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            for it in range(40):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((target_rank, p))
                        for k in range(p):
                            M[:, k] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, T[i,j,:], rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((target_rank, n))
                        for i in range(n):
                            M[:, i] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(M.T, T[:,j,k], rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((target_rank, m))
                        for j in range(m):
                            M[:, j] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(M.T, T[i,:,k], rcond=None)[0]
                        except:
                            pass
                
                if it % 10 == 0:
                    e = err_fast(U, V, W)
                    if e < 1e-7:
                        U_r = round_to_half(U)
                        V_r = round_to_half(V)
                        W_r = round_to_half(W)
                        if err_fast(U_r, V_r, W_r) < 1e-7:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                        else:
                            best_rank = target_rank
                            best_U, best_V, best_W = U, V, W
                        found = True
                        break
                    if it > 25 and e > 2.0:
                        break
        
        if found:
            break
    
    # Final verification
    final_err = err_fast(best_U, best_V, best_W)
    if final_err > 1e-6:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_school
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
