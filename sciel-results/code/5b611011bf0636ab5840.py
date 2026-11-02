import numpy as np
import time

def construct():
    deadline = time.time() + 110
    n, m, p = 3, 4, 8
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    rank_sb = n * m * p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, atol=1e-10):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=atol)
    
    assert verify(U, V, W)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb
    
    for target_rank in [88]:
        if time.time() > deadline - 30:
            break
        
        for trial in range(2):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + trial)
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            for iteration in range(15):
                if time.time() > deadline - 5:
                    break
                
                K_u = np.zeros((m*p*p*n, target_rank))
                for r_idx in range(target_rank):
                    K_u[:, r_idx] = np.outer(V_r[:, r_idx], W_r[:, r_idx]).reshape(-1)
                for a in range(n*m):
                    sol, res, rank, sv = np.linalg.lstsq(K_u.T, T[a,:,:].reshape(-1), rcond=None)
                    U_r[a, :] = sol
                
                K_v = np.zeros((n*m*p*n, target_rank))
                for r_idx in range(target_rank):
                    K_v[:, r_idx] = np.outer(U_r[:, r_idx], W_r[:, r_idx]).reshape(-1)
                for b in range(m*p):
                    sol, res, rank, sv = np.linalg.lstsq(K_v.T, T[:,b,:].reshape(-1), rcond=None)
                    V_r[b, :] = sol
                
                K_w = np.zeros((n*m*m*p, target_rank))
                for r_idx in range(target_rank):
                    K_w[:, r_idx] = np.outer(U_r[:, r_idx], V_r[:, r_idx]).reshape(-1)
                for c in range(p*n):
                    sol, res, rank, sv = np.linalg.lstsq(K_w.T, T[:,:,c].reshape(-1), rcond=None)
                    W_r[c, :] = sol
            
            if verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
