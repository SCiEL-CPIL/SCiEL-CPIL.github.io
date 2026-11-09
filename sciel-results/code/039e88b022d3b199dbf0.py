import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook fallback
    rank_sb = n*m*p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    T_flat_a = T.reshape(nm, mp*pn)
    T_flat_b = T.reshape(mp, nm*pn)
    T_flat_c = T.reshape(pn, nm*mp)
    
    # Try ranks from 74 down
    for trial_rank in range(rank_sb - 1, 50, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + trial_rank * 10 + attempt)
            U = np.random.randn(nm, trial_rank) * 0.1
            V = np.random.randn(mp, trial_rank) * 0.1
            W = np.random.randn(pn, trial_rank) * 0.1
            
            converged = False
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                
                # ALS updates
                for r in range(trial_rank):
                    vw = (V[:,r] * W[:,r]).reshape(mp, pn).flatten()
                    d = np.dot(vw, vw)
                    if d > 1e-10:
                        U[:,r] = T_flat_a @ vw / d
                for r in range(trial_rank):
                    uw = (U[:,r] * W[:,r]).reshape(nm, pn).flatten()
                    d = np.dot(uw, uw)
                    if d > 1e-10:
                        V[:,r] = T_flat_b @ uw / d
                for r in range(trial_rank):
                    uv = (U[:,r] * V[:,r]).reshape(nm, mp).flatten()
                    d = np.dot(uv, uv)
                    if d > 1e-10:
                        W[:,r] = T_flat_c @ uv / d
                
                if it % 50 == 49:
                    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                    err = np.max(np.abs(recon - T))
                    if err < 1e-8:
                        found = True
                        best_rank = trial_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
            
            if found:
                break
        
        if found:
            break
    
    # Round to half-integers
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    
    recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
    if np.max(np.abs(recon - T)) < 1e-6:
        U_out, V_out, W_out = U_r, V_r, W_r
    else:
        U_out, V_out, W_out = best_U, best_V, best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
