"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 138
    
    sb_rank = n * m * p
    
    U_sb = np.zeros((n*m, sb_rank))
    V_sb = np.zeros((m*p, sb_rank))
    W_sb = np.zeros((p*n, sb_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def residual_norm(U, V, W):
        r = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    total += (s - 1.0)**2
        return np.sqrt(total)
    
    def alt_min(U, V, W, max_iter=120, tol=1e-5):
        r = U.shape[1]
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)
                    U[i*m+j, :] = x
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-12)
                    V[j*p+k, :] = x
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)
                    W[k*n+i, :] = x
            if it % 20 == 0:
                rn = residual_norm(U, V, W)
                if rn < tol:
                    break
        return U, V, W, residual_norm(U, V, W)
    
    best_rank = sb_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try random restarts for various target ranks
    for target_rank in range(108, 119):
        if time.time() > deadline - 12:
            break
        
        found = False
        for restart in range(6):
            if time.time() > deadline - 8:
                break
            
            np.random.seed(1000 + target_rank * 100 + restart)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            U, V, W, rn = alt_min(U, V, W, max_iter=80)
            
            if rn < 1e-4:
                for denom in [1, 2, 4, 8]:
                    U_r = np.round(U * denom) / denom
                    V_r = np.round(V * denom) / denom
                    W_r = np.round(W * denom) / denom
                    rn_r = residual_norm(U_r, V_r, W_r)
                    if rn_r < 1e-6:
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                            found = True
                            break
        
        if found:
            break
    
    # Greedy column removal
    if time.time() < deadline - 10:
        U_cur = U_sb.copy()
        V_cur = V_sb.copy()
        W_cur = W_sb.copy()
        cur_rank = sb_rank
        
        for remove_idx in range(sb_rank):
            if time.time() > deadline - 5:
                break
            if remove_idx >= cur_rank:
                break
            
            new_rank = cur_rank - 1
            U_new = np.delete(U_cur, remove_idx, axis=1).copy()
            V_new = np.delete(V_cur, remove_idx, axis=1).copy()
            W_new = np.delete(W_cur, remove_idx, axis=1).copy()
            
            U_new, V_new, W_new, rn = alt_min(U_new, V_new, W_new, max_iter=15)
            
            if rn < 1e-3:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                cur_rank = new_rank
                if cur_rank < best_rank:
                    best_rank = cur_rank
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
        
        for denom in [1, 2, 4, 8]:
            U_r = np.round(best_U * denom) / denom
            V_r = np.round(best_V * denom) / denom
            W_r = np.round(best_W * denom) / denom
            rn = residual_norm(U_r, V_r, W_r)
            if rn < 1e-6:
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
