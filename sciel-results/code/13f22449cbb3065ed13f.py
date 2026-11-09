import time

DIMENSIONS = [2, 4, 5]

def construct():
    _t0 = time.time()
    _deadline = _t0 + 115.0  # 115s safety margin (evaluator kills at 150s)
    
    n, m, p = 2, 4, 5
    
    # We construct the decomposition using 2x2x2 Strassen blocks and 2x2x1 schoolbook blocks.
    # The problem 2x4x5 is split into two 2x2x5 sub-problems (rows of B split into two groups of 2).
    # Each 2x2x5 is split into one 2x2x2 and one 2x2x3.
    # 2x2x2 uses Strassen (rank 7).
    # 2x2x3 uses 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
    # Total rank per 2x2x5 block: 7 + 11 = 18.
    # Total rank: 18 + 18 = 36.
    
    rank = 36
    
    # Check deadline before heavy work
    if time.time() > _deadline - 5.0:
        # Fallback: return a valid rank-40 schoolbook decomposition
        U_fb = [[0]*40 for _ in range(n*m)]
        V_fb = [[0]*40 for _ in range(m*p)]
        W_fb = [[0]*40 for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen coefficients for 2x2x2
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # r=0..6
    S_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1]  # M7
    ]
    S_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [-1,0,1,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1]   # M7
    ]
    S_W = [
        [1,0,0,1],  # M1
        [0,1,0,-1], # M2
        [0,0,1,1],  # M3
        [1,1,0,0],  # M4
        [-1,0,1,0], # M5
        [0,0,0,1],  # M6
        [1,0,0,0]   # M7
    ]
    
    # Schoolbook coefficients for 2x2x1
    # U[i*2+j], V[j*2+0], W[0*2+i]
    # r=0..3
    SB_U = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    SB_V = [
        [1,0],
        [0,1],
        [1,0],
        [0,1]
    ]
    SB_W = [
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen_block(i_base, j_base, k_base, col_start):
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = S_U[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col_start + r] = S_V[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk * n + gi][col_start + r] = S_W[r][lk * 2 + li]
        return col_start + 7

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1_block(i_base, j_base, k_base, col_start):
        for r in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = SB_U[r][li * 2 + lj]
            for lj in range(2):
                gj = j_base + lj
                gk = k_base
                V[gj * p + gk][col_start + r] = SB_V[r][lj]
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk * n + gi][col_start + r] = SB_W[r][li]
        return col_start + 4

    # Block 1: j_base = 0
    col = add_strassen_block(0, 0, 0, col) # k=0,1
    col = add_strassen_block(0, 0, 2, col) # k=2,3
    col = add_schoolbook_2x2x1_block(0, 0, 4, col) # k=4
    
    # Block 2: j_base = 2
    col = add_strassen_block(0, 2, 0, col) # k=0,1
    col = add_strassen_block(0, 2, 2, col) # k=2,3
    col = add_schoolbook_2x2x1_block(0, 2, 4, col) # k=4
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    if time.time() > _deadline:
        # Fallback
        U_fb = [[0]*40 for _ in range(n*m)]
        V_fb = [[0]*40 for _ in range(m*p)]
        W_fb = [[0]*40 for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
