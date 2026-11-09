import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_2x2_strassen_factors():
    """
    Build U, V, W factors for 2x2 by 2x2 Strassen multiplication.
    U shape: (4, 7) for A indices (i,j) -> index i*2+j
    V shape: (4, 7) for B indices (j,k) -> index j*2+k
    W shape: (4, 7) for C indices (k,i) with trace convention W[k*2+i]
    
    Strassen for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 += M1, C22 += M1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 += M3, C22 += M3
    
    # M4 = A22(B12-B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C11 += M4, C21 += M4
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += M7
    
    return U, V, W

def _compose_4x4_strassen():
    """
    Compose 2x2 Strassen into 4x4 Strassen. Returns U (16,49), V (16,49), W (16,49).
    """
    U2, V2, W2 = _build_2x2_strassen_factors()
    
    s4x4 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    col = 0
    for t in range(7):
        A_blocks, B_blocks, C_blocks = s4x4[t]
        for s in range(7):
            for (br, bc), a_coeff in A_blocks.items():
                for li in range(2):
                    for lj in range(2):
                        u2 = U2[li*2+lj][s]
                        if u2 != 0:
                            gi = br*2 + li
                            gj = bc*2 + lj
                            U4[gi*4+gj][col] += a_coeff * u2
            for (bc, bk), b_coeff in B_blocks.items():
                for lj in range(2):
                    for lk in range(2):
                        v2 = V2[lj*2+lk][s]
                        if v2 != 0:
                            gj = bc*2 + lj
                            gk = bk*2 + lk
                            V4[gj*4+gk][col] += b_coeff * v2
            for (br, bk), c_coeff in C_blocks.items():
                for lk in range(2):
                    for li in range(2):
                        w2 = W2[lk*2+li][s]
                        if w2 != 0:
                            gk = bk*2 + lk
                            gi = br*2 + li
                            W4[gk*4+gi][col] += c_coeff * w2
            col += 1
    
    return U4, V4, W4

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute via decomposition
    # U term: for each r, u_r = sum_{i,j} U[i*m+j, r] * A[i,j]
    # V term: for each r, v_r = sum_{j,k} V[j*p+k, r] * B[j,k]
    # C[i,k] = sum_r W[k*n+i, r] * u_r * v_r
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_val = 0.0
        for i in range(n):
            for j in range(m):
                u_val += U_np[i*m+j, r] * A[i, j]
        v_val = 0.0
        for j in range(m):
            for k in range(p):
                v_val += V_np[j*p+k, r] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_np[k*n+i, r] * u_val * v_val
    
    error = np.max(np.abs(C_computed - C_expected))
    return error < 1e-10, error

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U4, V4, W4 = _compose_4x4_strassen()
    
    total_rank = 49 + 49 + 16
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(36)]
    W_full = [[0]*total_rank for _ in range(36)]
    
    # Part 1: A * B1 where B1 is columns 0-3 of B
    for r in range(49):
        for idx_u in range(16):
            if U4[idx_u][r] != 0:
                U_full[idx_u][r] = U4[idx_u][r]
        for idx_v in range(16):
            j = idx_v // 4
            k = idx_v % 4
            v_idx = j*9 + k
            if V4[idx_v][r] != 0:
                V_full[v_idx][r] = V4[idx_v][r]
        for idx_w in range(16):
            k = idx_w // 4
            i = idx_w % 4
            w_idx = k*4 + i
            if W4[idx_w][r] != 0:
                W_full[w_idx][r] = W4[idx_w][r]
    
    # Part 2: A * B2 where B2 is columns 4-7 of B
    offset2 = 49
    for r in range(49):
        for idx_u in range(16):
            if U4[idx_u][r] != 0:
                U_full[idx_u][r + offset2] = U4[idx_u][r]
        for idx_v in range(16):
            j = idx_v // 4
            k = idx_v % 4
            v_idx = j*9 + (k+4)
            if V4[idx_v][r] != 0:
                V_full[v_idx][r + offset2] = V4[idx_v][r]
        for idx_w in range(16):
            k = idx_w // 4
            i = idx_w % 4
            w_idx = (k+4)*4 + i
            if W4[idx_w][r] != 0:
                W_full[w_idx][r + offset2] = W4[idx_w][r]
    
    # Part 3: A * B3 where B3 is column 8 of B (4x1)
    offset3 = 98
    for i in range(4):
        for j in range(4):
            col = offset3 + i*4 + j
            U_full[i*4+j][col] = 1
            V_full[j*9+8][col] = 1
            W_full[8*4+i][col] = 1
    
    # Verify
    valid, error = _verify_decomposition(U_full, V_full, W_full, n, m, p)
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_full, "V": V_full, "W": W_full}
