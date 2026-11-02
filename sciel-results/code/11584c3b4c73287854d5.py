import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    def check_decomposition(U, V, W):
        T_hat = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_hat[i, j, k] = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
        return np.linalg.norm(T - T_hat)
    
    def round_to_half_int(arr):
        return np.round(arr * 2) / 2.0
    
    def verify_rounded(U, V, W):
        U_r = round_to_half_int(U)
        V_r = round_to_half_int(V)
        W_r = round_to_half_int(W)
        err = check_decomposition(U_r, V_r, W_r)
        return err < 1e-9, U_r, V_r, W_r
    
    def solve_rank(R, restarts=3):
        for restart in range(restarts):
            if time.time() - start_time > time_limit:
                return None
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(50):
                if time.time() - start_time > time_limit:
                    return None
                # Update U
                M = np.zeros((n * m * p, n * m * R))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                M[idx, (i * m + j) * R + r] = V[j * p + k, r] * W[k * n + i, r]
                            idx += 1
                t_vec = T.flatten()
                U_flat, _, _, _ = np.linalg.lstsq(M, t_vec, rcond=None)
                U = U_flat.reshape(n * m, R)
                
                # Update V
                M = np.zeros((n * m * p, m * p * R))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                M[idx, (j * p + k) * R + r] = U[i * m + j, r] * W[k * n + i, r]
                            idx += 1
                V_flat, _, _, _ = np.linalg.lstsq(M, t_vec, rcond=None)
                V = V_flat.reshape(m * p, R)
                
                # Update W
                M = np.zeros((n * m * p, p * n * R))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                M[idx, (k * n + i) * R + r] = U[i * m + j, r] * V[j * p + k, r]
                            idx += 1
                W_flat, _, _, _ = np.linalg.lstsq(M, t_vec, rcond=None)
                W = W_flat.reshape(p * n, R)
                
            err = check_decomposition(U, V, W)
            if err < 1e-6:
                valid, U_r, V_r, W_r = verify_rounded(U, V, W)
                if valid:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist(), "rank": R}
                    
        return None

    # Try ranks
    for R in range(99, 79, -1):
        if time.time() - start_time > time_limit - 10:
            break
        res = solve_rank(R, restarts=2)
        if res is not None:
            if res["rank"] < best_rank:
                best_rank = res["rank"]
                best_result = {"U": res["U"], "V": res["V"], "W": res["W"]}
                
    return best_result
