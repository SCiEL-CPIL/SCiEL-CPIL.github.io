import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def _build_rank77():
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=6, tol=1e-6):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

def _schoolbook(n=4, m=4, p=6):
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i*m + j) * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    return U, V, W

def _als_step(U, V, W, n, m, p, R):
    U_new = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            for k in range(p):
                for r in range(R):
                    A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
            U_new[i*m+j, :] = u
            
    V_new = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            for i in range(n):
                for r in range(R):
                    A[i, r] = U[i*m+j, r] * W[k*n+i, r]
            v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
            V_new[j*p+k, :] = v
            
    W_new = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, R))
            for j in range(m):
                for r in range(R):
                    A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W_new[k*n+i, :] = w
            
    return U_new, V_new, W_new

def _reduce_rank_als(U, V, W, n, m, p, max_iter=50, time_limit=10.0):
    R_orig = U.shape[1]
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_R = R_orig
    best_U, best_V, best_W = current_U, current_V, current_W
    
    start_time = time.time()
    
    for target_rank in range(R_orig - 1, max(0, R_orig - 10), -1):
        if time.time() - start_time > time_limit:
            break
            
        norms = np.linalg.norm(np.vstack([current_U, current_V, current_W]), axis=0)
        min_idx = np.argmin(norms)
        mask = np.ones(current_R, dtype=bool)
        mask[min_idx] = False
        
        U_active = current_U[:, mask]
        V_active = current_V[:, mask]
        W_active = current_W[:, mask]
        
        U_als, V_als, W_als = U_active, V_active, W_active
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            U_als, V_als, W_als = _als_step(U_als, V_als, W_als, n, m, p, target_rank)
            
        if _verify(U_als, V_als, W_als, n, m, p, tol=1e-5):
             U_rounded = np.round(U_als * 2) / 2
             V_rounded = np.round(V_als * 2) / 2
             W_rounded = np.round(W_als * 2) / 2
             if _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-5):
                 current_U, current_V, current_W = U_rounded, V_rounded, W_rounded
                 current_R = target_rank
                 best_U, best_V, best_W = current_U, current_V, current_W
             else:
                 current_U, current_V, current_W = U_als, V_als, W_als
                 current_R = target_rank
                 best_U, best_V, best_W = current_U, current_V, current_W
                 
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    U, V, W = _build_rank77()
    if _verify(U, V, W, n, m, p):
        remaining_time = TIME_LIMIT - (time.time() - START_TIME) - 5
        if remaining_time > 0:
            U_red, V_red, W_red = _reduce_rank_als(U, V, W, n, m, p, time_limit=remaining_time)
            if _verify(U_red, V_red, W_red, n, m, p):
                return {"U": _to_list(U_red), "V": _to_list(V_red), "W": _to_list(W_red)}
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    U, V, W = _schoolbook(n, m, p)
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
