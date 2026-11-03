"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank below schoolbook 50.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # We'll build U, V, W as lists of columns
    # U has shape (n*m, rank) = (10, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (10, rank)
    
    U_cols = []  # Each column is a list of length n*m = 10
    V_cols = []  # Each column is a list of length m*p = 25
    W_cols = []  # Each column is a list of length p*n = 10
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a single rank-1 term to the decomposition."""
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
    
    def zeros(length):
        return [0] * length
    
    # Partition: A is 2x5, split into A1 (2x2) and A2 (2x3)
    # B is 5x5, split into B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    # C = A*B, C is 2x5
    # C[:, :2] = A1*B11 + A2*B21
    # C[:, 2:] = A1*B12 + A2*B22
    
    # Helper: Strassen decomposition for 2x2 * 2x2
    # Given A (2x2), B (2x2), compute C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+d)*(h-g)
    # M6 = (b-c)*(e+g)
    # M7 = (b-d)*(f+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M4
    
    def strassen_2x2(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx):
        """
        Add Strassen terms for 2x2*2x2.
        a_idx, b_idx, c_idx, d_idx: indices in U for A's entries (row-major: a=top-left, b=top-right, c=bottom-left, d=bottom-right)
        e_idx, f_idx, g_idx, h_idx: indices in V for B's entries
        c_idx, etc. for output: we need to know which output positions these contribute to.
        """
        # For a 2x2*2x2 multiplication contributing to output positions,
        # we need to map the 4 output entries to their positions in the full output.
        # This function will be called with appropriate index mappings.
        pass
    
    # Let me think differently. I'll directly construct the terms.
    # 
    # Full problem: A (2x5) * B (5x5) = C (2x5)
    # A indices: A[i][j] for i in 0..1, j in 0..4
    # B indices: B[j][k] for j in 0..4, k in 0..4
    # C indices: C[i][k] for i in 0..1, k in 0..4
    #
    # U index: i*m + j = i*5 + j, for i in 0..1, j in 0..4 → 0..9
    # V index: j*p + k = j*5 + k, for j in 0..4, k in 0..4 → 0..24
    # W index: k*n + i = k*2 + i, for k in 0..4, i in 0..1 → 0..9
    #
    # Partition:
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:5] (2x3)
    # B11 = B[0:2, 0:2] (2x2), B12 = B[0:2, 2:5] (2x3)
    # B21 = B[2:5, 0:2] (3x2), B22 = B[2:5, 2:5] (3x3)
    #
    # C[:, 0:2] = A1*B11 + A2*B21
    # C[:, 2:5] = A1*B12 + A2*B22
    
    # --- Term 1: A1 * B11 (2x2 * 2x2) using Strassen ---
    # A1[i][j] for i,j in 0..1 → U index: i*5 + j
    # B11[j][k] for j,k in 0..1 → V index: j*5 + k
    # Output: C[i][k] for i,k in 0..1 → W index: k*2 + i
    
    # Strassen for 2x2*2x2:
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    
    a00_u = 0*5 + 0  # 0
    a01_u = 0*5 + 1  # 1
    a10_u = 1*5 + 0  # 5
    a11_u = 1*5 + 1  # 6
    
    b00_v = 0*5 + 0  # 0
    b01_v = 0*5 + 1  # 1
    b10_v = 1*5 + 0  # 5
    b11_v = 1*5 + 1  # 6
    
    c00_w = 0*2 + 0  # 0  (C[0][0])
    c01_w = 1*2 + 0  # 2  (C[0][1])
    c10_w = 0*2 + 1  # 1  (C[1][0])
    c11_w = 1*2 + 1  # 3  (C[1][1])
    
    # M1 = (a00+a01) * (b00+b11)
    u1 = zeros(10); u1[a00_u] = 1; u1[a01_u] = 1
    v1 = zeros(25); v1[b00_v] = 1; v1[b11_v] = 1
    w1 = zeros(10); w1[c00_w] = 1; w1[c01_w] = 1; w1[c10_w] = 1; w1[c11_w] = -1
    add_term(u1, v1, w1)
    
    # M2 = (a10+a11) * b00
    u2 = zeros(10); u2[a10_u] = 1; u2[a11_u] = 1
    v2 = zeros(25); v2[b00_v] = 1
    w2 = zeros(10); w2[c10_w] = 1; w2[c11_w] = -1
    add_term(u2, v2, w2)
    
    # M3 = a00 * (b01-b11)
    u3 = zeros(10); u3[a00_u] = 1
    v3 = zeros(25); v3[b01_v] = 1; v3[b11_v] = -1
    w3 = zeros(10); w3[c01_w] = 1; w2[c11_w] = 0  # careful, reuse
    w3 = zeros(10); w3[c01_w] = 1; w3[c11_w] = 1
    add_term(u3, v3, w3)
    
    # M4 = a11 * (b10-b00)
    u4 = zeros(10); u4[a11_u] = 1
    v4 = zeros(25); v4[b10_v] = 1; v4[b00_v] = -1
    w4 = zeros(10); w4[c00_w] = 1; w4[c11_w] = 1
    add_term(u4, v4, w4)
    
    # M5 = (a00+a11) * (b11-b10)
    u5 = zeros(10); u5[a00_u] = 1; u5[a11_u] = 1
    v5 = zeros(25); v5[b11_v] = 1; v5[b10_v] = -1
    w5 = zeros(10); w5[c00_w] = -1; w5[c01_w] = 1; w5[c10_w] = 1
    add_term(u5, v5, w5)
    
    # M6 = (a01-a10) * (b00+b10)
    u6 = zeros(10); u6[a01_u] = 1; u6[a10_u] = -1
    v6 = zeros(25); v6[b00_v] = 1; v6[b10_v] = 1
    w6 = zeros(10); w6[c00_w] = 1; w6[c10_w] = -1
    add_term(u6, v6, w6)
    
    # M7 = (a01-a11) * (b01+b11)
    u7 = zeros(10); u7[a01_u] = 1; u7[a11_u] = -1
    v7 = zeros(25); v7[b01_v] = 1; v7[b11_v] = 1
    w7 = zeros(10); w7[c00_w] = 1; w7[c01_w] = -1
    add_term(u7, v7, w7)
    
    # --- Term 2: A2 * B21 (2x3 * 3x2) ---
    # A2[i][j] for i in 0..1, j in 2..4 → U index: i*5 + j
    # B21[j][k] for j in 2..4, k in 0..1 → V index: j*5 + k
    # Output: C[i][k] for i in 0..1, k in 0..1 → W index: k*2 + i
    #
    # Split middle dimension 3 into 1+2:
    # A2 = [A2_1, A2_2] where A2_1 is 2x1 (j=2), A2_2 is 2x2 (j=3,4)
    # B21 = [B21_1; B21_2] where B21_1 is 1x2 (j=2), B21_2 is 2x2 (j=3,4)
    # A2*B21 = A2_1*B21_1 + A2_2*B21_2
    #
    # A2_1*B21_1: 2x1 * 1x2, rank 2
    # A2_2*B21_2: 2x2 * 2x2, Strassen rank 7
    # Total: 2 + 7 = 9
    
    # A2_1: A[i][2] for i in 0..1 → U index: i*5 + 2
    # B21_1: B[2][k] for k in 0..1 → V index: 2*5 + k
    # Output: C[i][k] for i,k in 0..1 → W index: k*2 + i
    
    # Term 2a: A2_1 * B21_1 (2x1 * 1x2)
    # This is sum over the single middle index j=2: A[i][2] * B[2][k]
    # Rank 2: for each (i,k) pair, but it's actually rank 2 since it's outer product structure
    # A2_1 is 2x1, B21_1 is 1x2, so A2_1*B21_1 is 2x2 with rank ≤ 2
    # Decomposition: for i in 0..1, k in 0..1: C[i][k] += A[i][2] * B[2][k]
    # This is 4 terms of rank 1, but we can do it in 2 terms:
    # Actually, 2x1 * 1x2 has bilinear complexity 2 (not 4)
    # C[i][k] = A[i][2] * B[2][k]
    # We can write this as: for each i, C[i][0] += A[i][2]*B[2][0], C[i][1] += A[i][2]*B[2][1]
    # That's 4 rank-1 terms. But the minimal rank for 2x1x1x2 is 2.
    # Let me just use 4 terms for simplicity, or find a 2-term decomposition.
    # 2x1 * 1x2: A is 2x1, B is 1x2. C[i][k] = A[i] * B[k].
    # This is an outer product, rank 1! Wait, no. A[i]*B[k] for i=0,1 and k=0,1.
    # C = [[A[0]*B[0], A[0]*B[1]], [A[1]*B[0], A[1]*B[1]]]
    # This has rank 1 as a matrix, but as a bilinear form, the tensor rank is 2.
    # Actually, the tensor for 2x1x1x2 multiplication has rank 2.
    # T[i,j,k,l] = delta(j,0)*delta(l,1) ... let me think.
    # The multiplication is C[i][k] = sum_j A[i][j] B[j][k] with j in {2} only (since it's 1-dim middle)
    # So C[i][k] = A[i][2] * B[2][k]
    # The tensor is T[i, j, k] where j is the middle index (only j=2), i is row, k is col.
    # T[i, 2, k] = 1 for all i, k. This is a 2x1x2 tensor with all entries 1.
    # The rank of this tensor is 2 (since it's the outer product of a 2-vector and a 2-vector).
    # So we need 2 terms:
    # Term: U has A[i][2] for i=0,1; V has B[2][0]; W has C[0][0] and C[1][0]
    # Term: U has A[i][2] for i=0,1; V has B[2][1]; W has C[0][1] and C[1][1]
    
    u8 = zeros(10); u8[0*5+2] = 1; u8[1*5+2] = 1
    v8 = zeros(25); v8[2*5+0] = 1
    w8 = zeros(10); w8[0*2+0] = 1; w8[0*2+1] = 1  # C[0][0], C[1][0]
    add_term(u8, v8, w8)
    
    u9 = zeros(10); u9[0*5+2] = 1; u9[1*5+2] = 1
    v9 = zeros(25); v9[2*5+1] = 1
    w9 = zeros(10); w9[1*2+0] = 1; w9[1*2+1] = 1  # C[0][1], C[1][1]
    add_term(u9, v9, w9)
    
    # A2_2*B21_2: 2x2 * 2x2 using Strassen
    # A2_2: A[i][3], A[i][4] for i in 0..1 → U index: i*5 + 3, i*5 + 4
    # B21_2: B[3][k], B[4][k] for k in 0..1 → V index: 3*5+k, 4*5+k
    # Output: C[i][k] for i,k in 0..1 → W index: k*2 + i
    
    a20_u = 0*5 + 3  # A[0][3]
    a21_u = 0*5 + 4  # A[0][4]
    a22_u = 1*5 + 3  # A[1][3]
    a23_u = 1*5 + 4  # A[1][4]
    
    b20_v = 3*5 + 0  # B[3][0]
    b21_v = 3*5 + 1  # B[3][1]
    b22_v = 4*5 + 0  # B[4][0]
    b23_v = 4*5 + 1  # B[4][1]
    
    # Strassen terms for A2_2 * B21_2
    # M1' = (A[0][3]+A[0][4]) * (B[3][0]+B[4][1])
    u10 = zeros(10); u10[a20_u] = 1; u10[a21_u] = 1
    v10 = zeros(25); v10[b20_v] = 1; v10[b23_v] = 1
    w10 = zeros(10); w10[c00_w] = 1; w10[c01_w] = 1; w10[c10_w] = 1; w10[c11_w] = -1
    add_term(u10, v10, w10)
    
    # M2' = (A[1][3]+A[1][4]) * B[3][0]
    u11 = zeros(10); u11[a22_u] = 1; u11[a23_u] = 1
    v11 = zeros(25); v11[b20_v] = 1
    w11 = zeros(10); w11[c10_w] = 1; w11[c11_w] = -1
    add_term(u11, v11, w11)
    
    # M3' = A[0][3] * (B[3][1]-B[4][1])
    u12 = zeros(10); u12[a20_u] = 1
    v12 = zeros(25); v12[b21_v] = 1; v12[b23_v] = -1
    w12 = zeros(10); w12[c01_w] = 1; w12[c11_w] = 1
    add_term(u12, v12, w12)
    
    # M4' = A[1][4] * (B[4][0]-B[3][0])
    u13 = zeros(10); u13[a23_u] = 1
    v13 = zeros(25); v13[b22_v] = 1; v13[b20_v] = -1
    w13 = zeros(10); w13[c00_w] = 1; w13[c11_w] = 1
    add_term(u13, v13, w13)
    
    # M5' = (A[0][3]+A[1][4]) * (B[4][1]-B[4][0])
    u14 = zeros(10); u14[a20_u] = 1; u14[a23_u] = 1
    v14 = zeros(25); v14[b23_v] = 1; v14[b22_v] = -1
    w14 = zeros(10); w14[c00_w] = -1; w14[c01_w] = 1; w14[c10_w] = 1
    add_term(u14, v14, w14)
    
    # M6' = (A[0][4]-A[1][3]) * (B[3][0]+B[4][0])
    u15 = zeros(10); u15[a21_u] = 1; u15[a22_u] = -1
    v15 = zeros(25); v15[b20_v] = 1; v15[b22_v] = 1
    w15 = zeros(10); w15[c00_w] = 1; w15[c10_w] = -1
    add_term(u15, v15, w15)
    
    # M7' = (A[0][4]-A[1][4]) * (B[3][1]+B[4][1])
    u16 = zeros(10); u16[a21_u] = 1; u16[a23_u] = -1
    v16 = zeros(25); v16[b21_v] = 1; v16[b23_v] = 1
    w16 = zeros(10); w16[c00_w] = 1; w16[c01_w] = -1
    add_term(u16, v16, w16)
    
    # --- Term 3: A1 * B12 (2x2 * 2x3) ---
    # A1: A[i][j] for i in 0..1, j in 0..1 → U index: i*5 + j
    # B12: B[j][k] for j in 0..1, k in 2..4 → V index: j*5 + k
    # Output: C[i][k] for i in 0..1, k in 2..4 → W index: k*2 + i
    #
    # Split middle dimension 2 into 1+1:
    # A1 = [A1_1, A1_2] where A1_1 is 2x1 (j=0), A1_2 is 2x1 (j=1)
    # B12 = [B12_1; B12_2] where B12_1 is 1x3 (j=0), B12_2 is 1x3 (j=1)
    # A1*B12 = A1_1*B12_1 + A1_2*B12_2
    # Each is 2x1 * 1x3, rank 2
    # Total: 4
    
    # A1_1 * B12_1: 2x1 * 1x3
    # A[i][0] * B[0][k] for k in 2..4
    # C[i][k] += A[i][0] * B[0][k]
    # Rank 2: for each k, one term
    for k in range(2, 5):
        u = zeros(10); u[0*5+0] = 1; u[1*5+0] = 1
        v = zeros(25); v[0*5+k] = 1
        w = zeros(10); w[k*2+0] = 1; w[k*2+1] = 1
        add_term(u, v, w)
    
    # A1_2 * B12_2: 2x1 * 1x3
    # A[i][1] * B[1][k] for k in 2..4
    for k in range(2, 5):
        u = zeros(10); u[0*5+1] = 1; u[1*5+1] = 1
        v = zeros(25); v[1*5+k] = 1
        w = zeros(10); w[k*2+0] = 1; w[k*2+1] = 1
        add_term(u, v, w)
    
    # --- Term 4: A2 * B22 (2x3 * 3x3) ---
    # A2: A[i][j] for i in 0..1, j in 2..4 → U index: i*5 + j
    # B22: B[j][k] for j in 2..4, k in 2..4 → V index: j*5 + k
    # Output: C[i][k] for i in 0..1, k in 2..4 → W index: k*2 + i
    #
    # Split middle dimension 3 into 1+2:
    # A2 = [A2_1, A2_2] where A2_1 is 2x1 (j=2), A2_2 is 2x2 (j=3,4)
    # B22 = [B22_1; B22_2] where B22_1 is 1x3 (j=2), B22_2 is 2x3 (j=3,4)
    # A2*B22 = A2_1*B22_1 + A2_2*B22_2
    # A2_1*B22_1: 2x1 * 1x3, rank 2
    # A2_2*B22_2: 2x2 * 2x3
    #   Split 2x3 into 2x1 + 2x2:
    #   A2_2*B22_2 = A2_2*B22_2a + A2_2*B22_2b where B22_2a is 2x1 (k=2), B22_2b is 2x2 (k=3,4)
    #   A2_2*B22_2a: 2x2 * 2x1, rank 4
    #   A2_2*B22_2b: 2x2 * 2x2, Strassen rank 7
    #   Total for A2_2*B22_2: 4 + 7 = 11
    # Total for A2*B22: 2 + 11 = 13
    
    # A2_1 * B22_1: 2x1 * 1x3
    # A[i][2] * B[2][k] for k in 2..4
    for k in range(2, 5):
        u = zeros(10); u[0*5+2] = 1; u[1*5+2] = 1
        v = zeros(25); v[2*5+k] = 1
        w = zeros(10); w[k*2+0] = 1; w[k*2+1] = 1
        add_term(u, v, w)
    
    # A2_2 * B22_2a: 2x2 * 2x1
    # A2_2: A[i][3], A[i][4] for i in 0..1
    # B22_2a: B[3][2], B[4][2]
    # Output: C[i][2] for i in 0..1
    # C[i][2] = A[i][3]*B[3][2] + A[i][4]*B[4][2]
    # This is 2x2 * 2x1, rank 4
    for i in range(2):
        for j_idx, jj in enumerate([3, 4]):
            u = zeros(10); u[i*5+jj] = 1
            v = zeros(25); v[jj*5+2] = 1
            w = zeros(10); w[2*2+i] = 1
            add_term(u, v, w)
    
    # A2_2 * B22_2b: 2x2 * 2x2 using Strassen
    # A2_2: A[i][3], A[i][4] for i in 0..1
    # B22_2b: B[3][3], B[3][4], B[4][3], B[4][4]
    # Output: C[i][3], C[i][4] for i in 0..1 → W index: 3*2+i, 4*2+i
    
    a30_u = 0*5 + 3  # A[0][3]
    a31_u = 0*5 + 4  # A[0][4]
    a32_u = 1*5 + 3  # A[1][3]
    a33_u = 1*5 + 4  # A[1][4]
    
    b30_v = 3*5 + 3  # B[3][3]
    b31_v = 3*5 + 4  # B[3][4]
    b32_v = 4*5 + 3  # B[4][3]
    b33_v = 4*5 + 4  # B[4][4]
    
    c30_w = 3*2 + 0  # C[0][3]
    c31_w = 4*2 + 0  # C[0][4]
    c32_w = 3*2 + 1  # C[1][3]
    c33_w = 4*2 + 1  # C[1][4]
    
    # Strassen for A2_2 * B22_2b
    # M1'' = (A[0][3]+A[0][4]) * (B[3][3]+B[4][4])
    u17 = zeros(10); u17[a30_u] = 1; u17[a31_u] = 1
    v17 = zeros(25); v17[b30_v] = 1; v17[b33_v] = 1
    w17 = zeros(10); w17[c30_w] = 1; w17[c31_w] = 1; w17[c32_w] = 1; w17[c33_w] = -1
    add_term(u17, v17, w17)
    
    # M2'' = (A[1][3]+A[1][4]) * B[3][3]
    u18 = zeros(10); u18[a32_u] = 1; u18[a33_u] = 1
    v18 = zeros(25); v18[b30_v] = 1
    w18 = zeros(10); w18[c32_w] = 1; w18[c33_w] = -1
    add_term(u18, v18, w18)
    
    # M3'' = A[0][3] * (B[3][4]-B[4][4])
    u19 = zeros(10); u19[a30_u] = 1
    v19 = zeros(25); v19[b31_v] = 1; v19[b33_v] = -1
    w19 = zeros(10); w19[c31_w] = 1; w19[c33_w] = 1
    add_term(u19, v19, w19)
    
    # M4'' = A[1][4] * (B[4][3]-B[3][3])
    u20 = zeros(10); u20[a33_u] = 1
    v20 = zeros(25); v20[b32_v] = 1; v20[b30_v] = -1
    w20 = zeros(10); w20[c30_w] = 1; w20[c33_w] = 1
    add_term(u20, v20, w20)
    
    # M5'' = (A[0][3]+A[1][4]) * (B[4][4]-B[4][3])
    u21 = zeros(10); u21[a30_u] = 1; u21[a33_u] = 1
    v21 = zeros(25); v21[b33_v] = 1; v21[b32_v] = -1
    w21 = zeros(10); w21[c30_w] = -1; w21[c31_w] = 1; w21[c32_w] = 1
    add_term(u21, v21, w21)
    
    # M6'' = (A[0][4]-A[1][3]) * (B[3][3]+B[4][3])
    u22 = zeros(10); u22[a31_u] = 1; u22[a32_u] = -1
    v22 = zeros(25); v22[b30_v] = 1; v22[b32_v] = 1
    w22 = zeros(10); w22[c30_w] = 1; w22[c32_w] = -1
    add_term(u22, v22, w22)
    
    # M7'' = (A[0][4]-A[1][4]) * (B[3][4]+B[4][4])
    u23 = zeros(10); u23[a31_u] = 1; u23[a33_u] = -1
    v23 = zeros(25); v23[b31_v] = 1; v23[b33_v] = 1
    w23 = zeros(10); w23[c30_w] = 1; w23[c31_w] = -1
    add_term(u23, v23, w23)
    
    # Build the final matrices
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(10)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(25)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(10)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

