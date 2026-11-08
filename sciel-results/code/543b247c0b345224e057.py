"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_mult_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _try_decomp(n, m, p, target_rank, time_limit, seed=0):
    T = _build_mult_tensor(n, m, p)
    T_mat = T.reshape(n*m, m*p * p*n)
    np.random.seed(seed)
    
    U = np.random.randn(n*m, target_rank) * 0.5
    V = np.random.randn(m*p, target_rank) * 0.5
    W = np.random.randn(p*n, target_rank) * 0.5
    
    for it in range(300):
        if time.time() > time_limit:
            break
        M = V[:, :, None] * W[None, :, :]
        M_mat = M.transpose(2, 0, 1).reshape(target_rank, -1)
        U = T_mat @ np.linalg.pinv(M_mat)
        
        N = U[:, :, None] * W[None, :, :]
        N_mat = N.transpose(2, 0, 1).reshape(target_rank, -1)
        T_b = T.transpose(1, 0, 2).reshape(m*p, -1)
        V = np.linalg.pinv(N_mat) @ T_b.T
        
        P = U[:, :, None] * V[None, :, :]
        P_mat = P.transpose(2, 0, 1).reshape(target_rank, -1)
        T_c = T.transpose(2, 0, 1).reshape(p*n, -1)
        W = np.linalg.pinv(P_mat) @ T_c.T
        
        T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
        err = np.linalg.norm(T_rec - T) / (np.linalg.norm(T) + 1e-15)
        if err < 1e-12:
            break
    
    T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
    err = np.linalg.norm(T_rec - T) / (np.linalg.norm(T) + 1e-15)
    if err < 1e-8:
        Ur = np.round(U * 2) / 2
        Vr = np.round(V * 2) / 2
        Wr = np.round(W * 2) / 2
        Tr = np.einsum('ad,bd,cd->abc', Ur, Vr, Wr)
        err_r = np.linalg.norm(Tr - T) / (np.linalg.norm(T) + 1e-15)
        if err_r < 1e-8:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    for target_rank in [44, 40, 36]:
        if time.time() > deadline:
            break
        for seed in range(100):
            if time.time() > deadline:
                break
            result = _try_decomp(n, m, p, target_rank, deadline, seed)
            if result is not None:
                U_l, V_l, W_l = result
                if target_rank < best["rank"]:
                    best = {"U": U_l, "V": V_l, "W": W_l, "rank": target_rank}
                break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
