import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 16, 28, 28
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute reshaped T matrices
    T_mat_w = T.transpose(2,0,1).reshape(pn, nm*mp)  # [c, (a,b)]
    T_mat_v = T.transpose(1,0,2).reshape(mp, nm*pn)  # [b, (a,c)]
    T_mat_u = T.transpose(0,1,2).reshape(nm, mp*pn)  # [a, (b,c)]
    
    def check(U, V, W):
        Trec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(Trec - T))
    
    def als_iter(U, V, W, rank):
        # Update W
        M = np.einsum('ar,br->abr', U, V).reshape(nm*mp, rank)
        Wn = np.linalg.lstsq(M, T_mat_w.T, rcond=None)[0].T
        
        # Update V
        N = np.einsum('ar,cr->acr', U, Wn).reshape(nm*pn, rank)
        Vn = np.linalg.lstsq(N, T_mat_v.T, rcond=None)[0].T
        
        # Update U
        P = np.einsum('br,cr->bcr', Vn, Wn).reshape(mp*pn, rank)
        Un = np.linalg.lstsq(P, T_mat_u.T, rcond=None)[0].T
        
        return Un, Vn, Wn
    
    # Schoolbook fallback
    rank_sb = n * m * p  # 112
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, c] = 1
                V_sb[j*p+k, c] = 1
                W_sb[k*n+i, c] = 1
                c += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    for rank in [95, 90, 85, 80, 75]:
        if time.time() > deadline:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            np.random.seed(seed * 1000 + rank * 7)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            for it in range(20):
                if time.time() > deadline:
                    break
                U, V, W = als_iter(U, V, W, rank)
            
            res = check(U, V, W)
            if res < 1e-6:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                found = True
                break
        
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
