import time

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U)>0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 105
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Fallback
    fb = _schoolbook(n, m, p)
    if not _check(fb["U"], fb["V"], fb["W"], n, m, p):
        return fb
    
    # Rank-42 solution using 6 blocks of Strassen (rank 7 each)
    # Split 12 columns into 6 pairs of 2
    n_blocks = p // 2  # 6 blocks
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # U_s: 4x7, rows [A00, A01, A10, A11]
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 0, 0, 1, 0, 0, -1]   # A11
    ]
    # V_s: 4x7, rows [B00, B01, B10, B11]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 0, 1, 0, 0, 1],   # B01
        [0, 0, 1, 0, 0, 1, 0],   # B10
        [1, 0, -1, 0, 1, 0, 0]   # B11
    ]
    # W_s: 4x7, rows [C00, C01, C10, C11]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 1, 0, 1, 0, 0],   # C01
        [0, 0, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    rank = n_blocks * 7  # 42
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 42
    
    for b in range(n_blocks):
        base = b * 7
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # U: same for all blocks, rows are A00, A01, A10, A11
        for i in range(4):
            for r in range(7):
                U[i][base + r] = float(U_s[i][r])
        
        # V: rows are B[j,k] for j in {0,1}, k in {k0, k1}
        # V_s row 0 (B00) -> V[j=0, k=k0] = V[0*p + k0] = V[k0]
        # V_s row 1 (B01) -> V[j=0, k=k1] = V[0*p + k1] = V[k1]
        # V_s row 2 (B10) -> V[j=1, k=k0] = V[1*p + k0] = V[p + k0]
        # V_s row 3 (B11) -> V[j=1, k=k1] = V[1*p + k1] = V[p + k1]
        v_row_map = [k0, k1, p + k0, p + k1]
        for vs_row in range(4):
            for r in range(7):
                V[v_row_map[vs_row]][base + r] = float(V_s[vs_row][r])
        
        # W: rows are C[k,i] indexed as k*n+i
        # W_s row 0 (C00) -> W[k=k0, i=0] = W[k0*n + 0]
        # W_s row 1 (C01) -> W[k=k0, i=1] = W[k0*n + 1]
        # W_s row 2 (C10) -> W[k=k1, i=0] = W[k1*n + 0]
        # W_s row 3 (C11) -> W[k=k1, i=1] = W[k1*n + 1]
        w_row_map = [k0 * n, k0 * n + 1, k1 * n, k1 * n + 1]
        for ws_row in range(4):
            for r in range(7):
                W[w_row_map[ws_row]][base + r] = float(W_s[ws_row][r])
    
    # Verify
    if _check(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return fb

# EVOLVE-BLOCK-END
