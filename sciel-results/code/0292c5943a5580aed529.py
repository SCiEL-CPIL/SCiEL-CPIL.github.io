import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    time_limit = 110.0  # Safety margin

    min_rank = max(n * m, m * p, p * n)
    
    best_rank = rank_schoolbook
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()

    num_restarts = 20
    max_iter = 100
    
    T_flat = T.reshape(n * m, m * p * p * n)
    T_flat_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_flat_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    for R in range(min_rank, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                M_flats = np.zeros((R, m * p * p * n))
                for r in range(R):
                    M_r = np.outer(V[:, r], W[:, r])
                    M_flats[r, :] = M_r.flatten()
                U_new, _, _, _ = np.linalg.lstsq(M_flats, T_flat.T, rcond=None)
                U = U_new.T
                
                # Update V
                M_flats_V = np.zeros((R, n * m * p * n))
                for r in range(R):
                    M_r = np.outer(U[:, r], W[:, r])
                    M_flats_V[r, :] = M_r.flatten()
                V_new, _, _, _ = np.linalg.lstsq(M_flats_V, T_flat_V.T, rcond=None)
                V = V_new.T
                
                # Update W
                M_flats_W = np.zeros((R, n * m * m * p))
                for r in range(R):
                    M_r = np.outer(U[:, r], V[:, r])
                    M_flats_W[r, :] = M_r.flatten()
                W_new, _, _, _ = np.linalg.lstsq(M_flats_W, T_flat_W.T, rcond=None)
                W = W_new.T
                
                # Check residual
                T_est = np.einsum('ar,br,cr->abc', U, V, W)
                res = np.linalg.norm(T - T_est)
                
                if res < 1e-6:
                    U_rounded = np.round(2 * U) / 2.0
                    V_rounded = np.round(2 * V) / 2.0
                    W_rounded = np.round(2 * W) / 2.0
                    
                    T_est_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    res_rounded = np.linalg.norm(T - T_est_rounded)
                    
                    if res_rounded < 1e-6:
                        best_rank = R
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        found = True
                        break

            if found:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
