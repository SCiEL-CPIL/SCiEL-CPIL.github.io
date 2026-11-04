import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    A, B, C = n * m, m * p, p * n
    
    target_tensor = np.zeros((A, B, C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i * m + j, j * p + k, k * n + i] = 1.0

    rank_school = n * m * p
    U_school = np.zeros((A, rank_school))
    V_school = np.zeros((B, rank_school))
    W_school = np.zeros((C, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    start_time = time.time()
    time_limit = 110

    T_U = target_tensor.reshape(A, B * C)
    T_V = target_tensor.transpose(0, 2, 1).reshape(A * C, B)
    T_W = target_tensor.reshape(A * B, C)

    for rank in range(rank_school - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(A, rank)
            V = np.random.randn(B, rank)
            W = np.random.randn(C, rank)
            
            for _ in range(100):
                # Update U
                M = np.einsum('br,cr->rbc', V, W).reshape(rank, B * C)
                MMT = M.T @ M
                RHS = M.T @ T_U.T
                try:
                    U = np.linalg.solve(MMT, RHS).T
                except np.linalg.LinAlgError:
                    U = np.linalg.lstsq(M.T, T_U.T, rcond=None)[0].T
                    
                # Update V
                N = np.einsum('ar,cr->acr', U, W).reshape(A * C, rank)
                NNT = N @ N.T
                RHS = N @ T_V
                try:
                    V = np.linalg.solve(NNT, RHS)
                except np.linalg.LinAlgError:
                    V = np.linalg.lstsq(N, T_V, rcond=None)[0]
                    
                # Update W
                P = np.einsum('ar,br->abr', U, V).reshape(A * B, rank)
                PPT = P @ P.T
                RHS = P @ T_W
                try:
                    W = np.linalg.solve(PPT, RHS)
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(P, T_W, rcond=None)[0]
                    
                if _ % 20 == 0:
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    residual = np.linalg.norm(target_tensor - T_approx)
                    if residual < 1e-6:
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        T_approx_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                        residual_r = np.linalg.norm(target_tensor - T_approx_r)
                        if residual_r < 1e-9:
                            best_rank = rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                            found = True
                            break
            if found:
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
