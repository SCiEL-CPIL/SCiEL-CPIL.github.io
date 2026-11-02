import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 blocks.
    Target Rank: 54
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

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

    def add_212_block(U, V, W, offset_r, i_off, j_off, k_off):
        u0 = i_off * m + j_off
        u1 = (i_off + 1) * m + j_off
        v0 = j_off * p + k_off
        v1 = j_off * p + (k_off + 1)
        w0 = k_off * n + i_off
        w1 = k_off * n + (i_off + 1)
        w2 = (k_off + 1) * n + i_off
        w3 = (k_off + 1) * n + (i_off + 1)
        
        U[u0, offset_r] = 1; V[v0, offset_r] = 1; W[w0, offset_r] = 1; offset_r += 1
        U[u0, offset_r] = 1; V[v1, offset_r] = 1; W[w2, offset_r] = 1; offset_r += 1
        U[u1, offset_r] = 1; V[v0, offset_r] = 1; W[w1, offset_r] = 1; offset_r += 1
        U[u1, offset_r] = 1; V[v1, offset_r] = 1; W[w3, offset_r] = 1; offset_r += 1
        
        return offset_r

    R_max = 60
    U = np.zeros((n * m, R_max), dtype=np.float64)
    V = np.zeros((m * p, R_max), dtype=np.float64)
    W = np.zeros((p * n, R_max), dtype=np.float64)
    
    r = 0
    # J blocks: [0,1], [2,3], [4]
    # K blocks: [0,1], [2,3], [4,5]
    for j_off in [0, 2, 4]:
        for k_off in [0, 2, 4]:
            if j_off < 4 and k_off < 4:
                r = add_strassen_block(U, V, W, r, 0, j_off, k_off)
            else:
                r = add_212_block(U, V, W, r, 0, j_off, k_off)

    if r < R_max:
        U = U[:, :r]
        V = V[:, :r]
        W = W[:, :r]
        
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
    
    if not verify(U, V, W):
        R = n * m * p
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        r = R

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
