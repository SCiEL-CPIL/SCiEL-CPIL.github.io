import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def verify(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_exp[a, b, c] = 1.0
        return np.allclose(T, T_exp, atol=1e-9)
    
    def add_strassen_block(U, V, W, offset_r, i_off, j_off, k_off):
        u_rows = [
            (i_off + 0) * m + (j_off + 0),
            (i_off + 0) * m + (j_off + 1),
            (i_off + 1) * m + (j_off + 0),
            (i_off + 1) * m + (j_off + 1)
        ]
        v_rows = [
            (j_off + 0) * p + (k_off + 0),
            (j_off + 0) * p + (k_off + 1),
            (j_off + 1) * p + (k_off + 0),
            (j_off + 1) * p + (k_off + 1)
        ]
        w_rows = [
            (k_off + 0) * n + (i_off + 0),
            (k_off + 0) * n + (i_off + 1),
            (k_off + 1) * n + (i_off + 0),
            (k_off + 1) * n + (i_off + 1)
        ]
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
        ]
        
        for u, v, w in terms:
            for idx in range(4):
                U[u_rows[idx], offset_r] += u[idx]
                V[v_rows[idx], offset_r] += v[idx]
                W[w_rows[idx], offset_r] += w[idx]
            offset_r += 1
        return offset_r
    
    def add_trivial_block(U, V, W, offset_r, i_off, j_off, k_off, ni, mj, kk):
        for i in range(ni):
            for j in range(mj):
                for k in range(kk):
                    a = (i_off + i) * m + (j_off + j)
                    b = (j_off + j) * p + (k_off + k)
                    c = (k_off + k) * n + (i_off + i)
                    U[a, offset_r] = 1.0
                    V[b, offset_r] = 1.0
                    W[c, offset_r] = 1.0
                    offset_r += 1
        return offset_r
    
    # Try multiple block decompositions
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Decomposition 1: Current (m=[2,2,1], p=[2,2,2])
    R_max = 70
    U = np.zeros((n*m, R_max), dtype=np.float64)
    V = np.zeros((m*p, R_max), dtype=np.float64)
    W = np.zeros((p*n, R_max), dtype=np.float64)
    r = 0
    for j_off, j_size in [(0,2), (2,2), (4,1)]:
        for k_off, k_size in [(0,2), (2,2), (4,2)]:
            if j_size == 2 and k_size == 2:
                r = add_strassen_block(U, V, W, r, 0, j_off, k_off)
            else:
                r = add_trivial_block(U, V, W, r, 0, j_off, k_off, 2, j_size, k_size)
    if r < best_rank and verify(U[:, :r], V[:, :r], W[:, :r]):
        best_rank = r
        best_U, best_V, best_W = U[:, :r].copy(), V[:, :r].copy(), W[:, :r].copy()
    
    if time.time() > deadline - 30:
        if best_U is not None:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        # Fallback to trivial
        R = n*m*p
        U = np.zeros((n*m, R), dtype=np.float64)
        V = np.zeros((m*p, R), dtype=np.float64)
        W = np.zeros((p*n, R), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Decomposition 2: m=[1,2,2], p=[2,2,2]
    U = np.zeros((n*m, R_max), dtype=np.float64)
    V = np.zeros((m*p, R_max), dtype=np.float64)
    W = np.zeros((p*n, R_max), dtype=np.float64)
    r = 0
    for j_off, j_size in [(0,1), (1,2), (3,2)]:
        for k_off, k_size in [(0,2), (2,2), (4,2)]:
            if j_size == 2 and k_size == 2:
                r = add_strassen_block(U, V, W, r, 0, j_off, k_off)
            else:
                r = add_trivial_block(U, V, W, r, 0, j_off, k_off, 2, j_size, k_size)
    if r < best_rank and verify(U[:, :r], V[:, :r], W[:, :r]):
        best_rank = r
        best_U, best_V, best_W = U[:, :r].copy(), V[:, :r].copy(), W[:, :r].copy()
    
    # Decomposition 3: m=[2,3], p=[3,3]
    U = np.zeros((n*m, R_max), dtype=np.float64)
    V = np.zeros((m*p, R_max), dtype=np.float64)
    W = np.zeros((p*n, R_max), dtype=np.float64)
    r = 0
    for j_off, j_size in [(0,2), (2,3)]:
        for k_off, k_size in [(0,3), (3,3)]:
            r = add_trivial_block(U, V, W, r, 0, j_off, k_off, 2, j_size, k_size)
    if r < best_rank and verify(U[:, :r], V[:, :r], W[:, :r]):
        best_rank = r
        best_U, best_V, best_W = U[:, :r].copy(), V[:, :r].copy(), W[:, :r].copy()
    
    # Decomposition 4: m=[3,2], p=[2,2,2]
    U = np.zeros((n*m, R_max), dtype=np.float64)
    V = np.zeros((m*p, R_max), dtype=np.float64)
    W = np.zeros((p*n, R_max), dtype=np.float64)
    r = 0
    for j_off, j_size in [(0,3), (3,2)]:
        for k_off, k_size in [(0,2), (2,2), (4,2)]:
            if j_size == 2 and k_size == 2:
                r = add_strassen_block(U, V, W, r, 0, j_off, k_off)
            else:
                r = add_trivial_block(U, V, W, r, 0, j_off, k_off, 2, j_size, k_size)
    if r < best_rank and verify(U[:, :r], V[:, :r], W[:, :r]):
        best_rank = r
        best_U, best_V, best_W = U[:, :r].copy(), V[:, :r].copy(), W[:, :r].copy()
    
    # Decomposition 5: m=[2,2,1], p=[3,3]
    U = np.zeros((n*m, R_max), dtype=np.float64)
    V = np.zeros((m*p, R_max), dtype=np.float64)
    W = np.zeros((p*n, R_max), dtype=np.float64)
    r = 0
    for j_off, j_size in [(0,2), (2,2), (4,1)]:
        for k_off, k_size in [(0,3), (3,3)]:
            r = add_trivial_block(U, V, W, r, 0, j_off, k_off, 2, j_size, k_size)
    if r < best_rank and verify(U[:, :r], V[:, :r], W[:, :r]):
        best_rank = r
        best_U, best_V, best_W = U[:, :r].copy(), V[:, :r].copy(), W[:, :r].copy()
    
    # Decomposition 6: m=[1,1,3], p=[2,2,2]
    U = np.zeros((n*m, R_max), dtype=np.float64)
    V = np.zeros((m*p, R_max), dtype=np.float64)
    W = np.zeros((p*n, R_max), dtype=np.float64)
    r = 0
    for j_off, j_size in [(0,1), (1,1), (2,3)]:
        for k_off, k_size in [(0,2), (2,2), (4,2)]:
            if j_size == 2 and k_size == 2:
                r = add_strassen_block(U, V, W, r, 0, j_off, k_off)
            else:
                r = add_trivial_block(U, V, W, r, 0, j_off, k_off, 2, j_size, k_size)
    if r < best_rank and verify(U[:, :r], V[:, :r], W[:, :r]):
        best_rank = r
        best_U, best_V, best_W = U[:, :r].copy(), V[:, :r].copy(), W[:, :r].copy()
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to trivial
    R = n*m*p
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
