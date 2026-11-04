import time
import numpy as np
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0
    
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def validate(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        s += U[ui, r] * V[vj, r] * W[wk, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def try_merge_terms(U, V, W, rank, a, b):
        target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] + U[i*m+j, b] * V[j*p+k, b] * W[k*n+i, b]
                    target[i,j,k] = val
        
        non_zero = np.argwhere(np.abs(target) > 1e-9)
        if len(non_zero) == 0:
            return None
        
        if len(non_zero) > 4:
            return None
        
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        values = [0, 1, -1, 0.5, -0.5]
        
        def backtrack(idx):
            if idx == len(non_zero):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = target[i,j,k]
                            actual = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            if abs(expected - actual) > 1e-9:
                                return False
                return True
            
            i,j,k = non_zero[idx]
            expected = target[i,j,k]
            ui_idx = i*m+j
            vj_idx = j*p+k
            wk_idx = k*n+i
            
            for uv in values:
                for vv in values:
                    for ww in values:
                        if abs(uv * vv * ww - expected) > 1e-9:
                            continue
                        if u[ui_idx] != 0 and abs(u[ui_idx] - uv) > 1e-9:
                            continue
                        if v[vj_idx] != 0 and abs(v[vj_idx] - vv) > 1e-9:
                            continue
                        if w[wk_idx] != 0 and abs(w[wk_idx] - ww) > 1e-9:
                            continue
                        
                        u[ui_idx] = uv
                        v[vj_idx] = vv
                        w[wk_idx] = ww
                        
                        if backtrack(idx + 1):
                            return True
                        
                        u[ui_idx] = 0
                        v[vj_idx] = 0
                        w[wk_idx] = 0
            
            return False
        
        if backtrack(0):
            return u, v, w
        
        return None

    U = U_sb.astype(float).copy()
    V = V_sb.astype(float).copy()
    W = W_sb.astype(float).copy()
    rank = base_rank
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(rank):
            if time.time() > deadline:
                break
            for b in range(a + 1, rank):
                if time.time() > deadline:
                    break
                result = try_merge_terms(U, V, W, rank, a, b)
                if result is not None:
                    u, v, w = result
                    new_U = np.zeros((n*m, rank-1))
                    new_V = np.zeros((m*p, rank-1))
                    new_W = np.zeros((p*n, rank-1))
                    
                    new_rank = 0
                    for r in range(rank):
                        if r != a and r != b:
                            new_U[:, new_rank] = U[:, r]
                            new_V[:, new_rank] = V[:, r]
                            new_W[:, new_rank] = W[:, r]
                            new_rank += 1
                    
                    new_U[:, new_rank] = u
                    new_V[:, new_rank] = v
                    new_W[:, new_rank] = w
                    new_rank += 1
                    
                    if validate(new_U, new_V, new_W, new_rank):
                        U, V, W = new_U, new_V, new_W
                        rank = new_rank
                        improved = True
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break

    if best_rank < rank:
        U, V, W, rank = best_U, best_V, best_W, best_rank

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
