import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T
    # T[u, v, w] = 1 if u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    schoolbook_rank = n * m * p
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    deadline = 145.0 # 5s margin

    # Pre-compute tensor slices for ALS
    T_W_const = T.reshape(n * m * m * p, p * n)
    T_U_const = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
    T_V_const = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)

    def check_solution(U, V, W):
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        err = np.linalg.norm(T_calc - T)
        return err < 1e-9

    def snap_to_half_int(U, V, W):
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        return U_int, V_int, W_int

    def solve_als_optimized(r, max_iters=20, restarts=5, init_U=None, init_V=None, init_W=None):
        for restart in range(restarts):
            if time.time() > deadline:
                return None
                
            if init_U is not None and restart == 0:
                if r == schoolbook_rank:
                    U = init_U.copy()
                    V = init_V.copy()
                    W = init_W.copy()
                else:
                    idx = np.random.choice(schoolbook_rank, r, replace=False)
                    U = init_U[:, idx].copy()
                    V = init_V[:, idx].copy()
                    W = init_W[:, idx].copy()
            else:
                U = np.random.uniform(-1, 1, (n * m, r))
                V = np.random.uniform(-1, 1, (m * p, r))
                W = np.random.uniform(-1, 1, (p * n, r))
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None

                # Update W
                A_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
                A_UV_flat = A_UV.reshape(n * m * m * p, r)
                
                try:
                    W_T, residuals, rank, sv = np.linalg.lstsq(A_UV_flat, T_W_const, rcond=None)
                    W = W_T.T
                except:
                    continue

                # Update U
                A_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
                A_VW_flat = A_VW.reshape(m * p * p * n, r)
                
                try:
                    U_T, residuals, rank, sv = np.linalg.lstsq(A_VW_flat, T_U_const, rcond=None)
                    U = U_T.T
                except:
                    continue

                # Update V
                A_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
                A_UW_flat = A_UW.reshape(n * m * p * n, r)
                
                try:
                    V_T, residuals, rank, sv = np.linalg.lstsq(A_UW_flat, T_V_const, rcond=None)
                    V = V_T.T
                except:
                    continue

                # Check error
                if it % 5 == 0:
                    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_calc - T)
                    if err < 1e-7:
                        U_int, V_int, W_int = snap_to_half_int(U, V, W)
                        if check_solution(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                        U_int2 = np.round(U)
                        V_int2 = np.round(V)
                        W_int2 = np.round(W)
                        if check_solution(U_int2, V_int2, W_int2):
                            return U_int2, V_int2, W_int2

        return None

    # Search from high rank to low rank
    ranks_to_try = list(range(schoolbook_rank, 50, -5))
    ranks_to_try += list(range(50, 25, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        result = solve_als_optimized(r, max_iters=20, restarts=5, init_U=U_sb, init_V=V_sb, init_W=W_sb)
        if result:
            U_int, V_int, W_int = result
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
        result = solve_als_optimized(r, max_iters=20, restarts=5)
        if result:
            U_int, V_int, W_int = result
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
