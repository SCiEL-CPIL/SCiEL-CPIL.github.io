import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, i*n+k] = 1.0
    
    rank_sch = n*m*p
    U_sch = np.zeros((n*m, rank_sch))
    V_sch = np.zeros((m*p, rank_sch))
    W_sch = np.zeros((n*p, rank_sch))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[i*n+k, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    best_rank = rank_sch
    
    # Try a few ranks with minimal ALS
    for target_rank in [70, 65, 60]:
        if time.time() > deadline - 30:
            break
        
        rng = np.random.default_rng(42)
        U = rng.standard_normal((n*m, target_rank))
        V = rng.standard_normal((m*p, target_rank))
        W = rng.standard_normal((n*p, target_rank))
        
        for it in range(10):
            if time.time() > deadline - 5:
                break
            M = (V[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
            U = np.linalg.lstsq(M.T, T.reshape(n*m, -1), rcond=None)[0]
            M = (U[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
            V = np.linalg.lstsq(M.T, T.transpose(1,0,2).reshape(m*p, -1), rcond=None)[0]
            M = (U[:, :, None] * V[None, :, :]).reshape(-1, target_rank)
            W = np.linalg.lstsq(M.T, T.transpose(2,0,1).reshape(n*p, -1), rcond=None)[0]
        
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        error = np.max(np.abs(recon - T))
        if error < 1e-2:
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result
