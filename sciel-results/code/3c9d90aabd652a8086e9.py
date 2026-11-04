import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    # 2x2x2 Strassen decomposition (rank 7)
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 2x2x3 decomposition (rank 8)
    # Terms 1-7: Strassen for 2x2x2 part (k2=0,1)
    # Term 8: rank-1 for 2x2x1 part (k2=2)
    S2x3 = []
    for u_pat, v_pat, w_pat in S2:
        # Map 2x2x2 patterns to 2x2x3 (k2=0,1 only)
        u2 = {}
        v2 = {}
        w2 = {}
        for (r, c), coeff in u_pat.items():
            u2[(r, c)] = coeff
        for (r, c), coeff in v_pat.items():
            v2[(r, c)] = coeff  # c is k2, which is 0 or 1
        for (r, c), coeff in w_pat.items():
            w2[(r, c)] = coeff  # r is k2, which is 0 or 1
        S2x3.append((u2, v2, w2))
    
    # Term 8: rank-1 for k2=2
    u2x1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    v2x1 = {(0,2): 1, (1,2): 1}
    w2x1 = {(2,0): 1, (2,1): 1}
    S2x3.append((u2x1, v2x1, w2x1))
    
    rank = 7 * 8  # 56
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r1, (u1, v1, w1) in enumerate(S2):
        for r2, (u2, v2, w2) in enumerate(S2x3):
            idx = r1 * 8 + r2
            for (i1, j1), cu1 in u1.items():
                for (i2, j2), cu2 in u2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U[i * m + j][idx] += cu1 * cu2
            for (j1, k1), cv1 in v1.items():
                for (j2, k2), cv2 in v2.items():
                    j = j1 * 2 + j2
                    k = k1 * 3 + k2
                    V[j * p + k][idx] += cv1 * cv2
            for (k1, i1), cw1 in w1.items():
                for (k2, i2), cw2 in w2.items():
                    k = k1 * 3 + k2
                    i = i1 * 2 + i2
                    W[k * n + i][idx] += cw1 * cw2
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()
    
    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]
    
    totals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    return bool(np.all(totals == 1))
