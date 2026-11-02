import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-22 decomposition.
    # This is based on partitioning 3x3 into 2x2 and 1x1 blocks.
    # The 2x2 part uses Strassen's algorithm (rank 7).
    # The cross terms and 1x1 part use schoolbook (rank 15).
    # Total rank 22.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    # u: list of length n*m (9)
    # v: list of length m*p (9)
    # w: list of length p*n (9)
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    def zero_vec(size):
        return [0] * size
        
    # Indices:
    # A[i, j] is at U index i*m + j = i*3 + j
    # B[j, k] is at V index j*p + k = j*3 + k
    # C[i, k] is at W index k*n + i = k*3 + i (Trace convention W[k*n+i])
    
    # 1. Strassen terms for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (A00 + A11) * (B00 + B11)
    u1 = zero_vec(9); u1[0] = 1; u1[3] = 1
    v1 = zero_vec(9); v1[0] = 1; v1[3] = 1
    w1 = zero_vec(9)
    # C00 += M1, C11 += M1
    w1[0] = 1 # k=0, i=0 -> 0*3+0=0
    w1[11] = 1 # k=2? No. C11 is i=1, k=1. W[1*3+1] = W[4].
    # Wait, W index is k*n + i.
    # C00: i=0, k=0. W[0].
    # C11: i=1, k=1. W[1*3+1] = W[4].
    # C01: i=0, k=1. W[1*3+0] = W[3].
    # C10: i=1, k=0. W[0*3+1] = W[1].
    
    # Let's re-verify W indices.
    # W[k*n + i]
    # C[i,k]
    # i=0, k=0 -> W[0]
    # i=0, k=1 -> W[3]
    # i=0, k=2 -> W[6]
    # i=1, k=0 -> W[1]
    # i=1, k=1 -> W[4]
    # i=1, k=2 -> W[7]
    # i=2, k=0 -> W[2]
    # i=2, k=1 -> W[5]
    # i=2, k=2 -> W[8]
    
    # Strassen M1: (A00+A11)(B00+B11)
    # Contributes to C00 (+1), C11 (+1)
    w1[0] = 1
    w1[4] = 1
    add_term(u1, v1, w1)
    
    # M2 = (A10+A11) * B00
    u2 = zero_vec(9); u2[3] = 1; u2[4] = 1
    v2 = zero_vec(9); v2[0] = 1
    w2 = zero_vec(9)
    # C10 (+1), C11 (-1)
    w2[1] = 1
    w2[4] = -1
    add_term(u2, v2, w2)
    
    # M3 = A00 * (B10 - B11)
    u3 = zero_vec(9); u3[0] = 1
    v3 = zero_vec(9); v3[3] = 1; v3[4] = -1
    w3 = zero_vec(9)
    # C01 (+1), C11 (+1)
    w3[3] = 1
    w3[4] = 1
    add_term(u3, v3, w3)
    
    # M4 = A11 * (B01 - B00)
    u4 = zero_vec(9); u4[4] = 1
    v4 = zero_vec(9); v4[1] = 1; v4[0] = -1
    w4 = zero_vec(9)
    # C00 (+1), C10 (+1)
    w4[0] = 1
    w4[1] = 1
    add_term(u4, v4, w4)
    
    # M5 = (A00+A01) * B11
    u5 = zero_vec(9); u5[0] = 1; u5[1] = 1
    v5 = zero_vec(9); v5[4] = 1
    w5 = zero_vec(9)
    # C00 (-1), C01 (+1)
    w5[0] = -1
    w5[3] = 1
    add_term(u5, v5, w5)
    
    # M6 = (A10-A00) * (B00+B01)
    u6 = zero_vec(9); u6[3] = 1; u6[0] = -1
    v6 = zero_vec(9); v6[0] = 1; v6[1] = 1
    w6 = zero_vec(9)
    # C11 (+1)
    w6[4] = 1
    add_term(u6, v6, w6)
    
    # M7 = (A01-A11) * (B10+B11)
    u7 = zero_vec(9); u7[1] = 1; u7[4] = -1
    v7 = zero_vec(9); v7[3] = 1; v7[4] = 1
    w7 = zero_vec(9)
    # C00 (+1), C10 (-1)
    w7[0] = 1
    w7[1] = -1
    add_term(u7, v7, w7)
    
    # 2. Cross terms A[0:2, 2] * B[2, 0:2]
    # A02 * B20 -> C00
    u8 = zero_vec(9); u8[2] = 1
    v8 = zero_vec(9); v8[6] = 1
    w8 = zero_vec(9); w8[0] = 1
    add_term(u8, v8, w8)
    
    # A02 * B21 -> C01
    u9 = zero_vec(9); u9[2] = 1
    v9 = zero_vec(9); v9[7] = 1
    w9 = zero_vec(9); w9[3] = 1
    add_term(u9, v9, w9)
    
    # A12 * B20 -> C10
    u10 = zero_vec(9); u10[5] = 1
    v10 = zero_vec(9); v10[6] = 1
    w10 = zero_vec(9); w10[1] = 1
    add_term(u10, v10, w10)
    
    # A12 * B21 -> C11
    u11 = zero_vec(9); u11[5] = 1
    v11 = zero_vec(9); v11[7] = 1
    w11 = zero_vec(9); w11[4] = 1
    add_term(u11, v11, w11)
    
    # 3. Cross terms A[0:2, 0:2] * B[0:2, 2]
    # A00 * B02 -> C02
    u12 = zero_vec(9); u12[0] = 1
    v12 = zero_vec(9); v12[2] = 1
    w12 = zero_vec(9); w12[6] = 1
    add_term(u12, v12, w12)
    
    # A01 * B12 -> C02
    u13 = zero_vec(9); u13[1] = 1
    v13 = zero_vec(9); v13[5] = 1
    w13 = zero_vec(9); w13[6] = 1
    add_term(u13, v13, w13)
    
    # A10 * B02 -> C12
    u14 = zero_vec(9); u14[3] = 1
    v14 = zero_vec(9); v14[2] = 1
    w14 = zero_vec(9); w14[7] = 1
    add_term(u14, v14, w14)
    
    # A11 * B12 -> C12
    u15 = zero_vec(9); u15[4] = 1
    v15 = zero_vec(9); v15[5] = 1
    w15 = zero_vec(9); w15[7] = 1
    add_term(u15, v15, w15)
    
    # 4. Cross terms A[2, 0:2] * B[0:2, 0:2]
    # A20 * B00 -> C20
    u16 = zero_vec(9); u16[6] = 1
    v16 = zero_vec(9); v16[0] = 1
    w16 = zero_vec(9); w16[2] = 1
    add_term(u16, v16, w16)
    
    # A21 * B10 -> C20
    u17 = zero_vec(9); u17[7] = 1
    v17 = zero_vec(9); v17[3] = 1
    w17 = zero_vec(9); w17[2] = 1
    add_term(u17, v17, w17)
    
    # A20 * B01 -> C21
    u18 = zero_vec(9); u18[6] = 1
    v18 = zero_vec(9); v18[1] = 1
    w18 = zero_vec(9); w18[5] = 1
    add_term(u18, v18, w18)
    
    # A21 * B11 -> C21
    u19 = zero_vec(9); u19[7] = 1
    v19 = zero_vec(9); v19[4] = 1
    w19 = zero_vec(9); w19[5] = 1
    add_term(u19, v19, w19)
    
    # 5. A[2, 0:2] * B[0:2, 2] and A[2,2]*B[2,2] -> C22
    # A20 * B02 -> C22
    u20 = zero_vec(9); u20[6] = 1
    v20 = zero_vec(9); v20[2] = 1
    w20 = zero_vec(9); w20[8] = 1
    add_term(u20, v20, w20)
    
    # A21 * B12 -> C22
    u21 = zero_vec(9); u21[7] = 1
    v21 = zero_vec(9); v21[5] = 1
    w21 = zero_vec(9); w21[8] = 1
    add_term(u21, v21, w21)
    
    # A22 * B22 -> C22
    u22 = zero_vec(9); u22[8] = 1
    v22 = zero_vec(9); v22[8] = 1
    w22 = zero_vec(9); w22[8] = 1
    add_term(u22, v22, w22)
    
    rank = len(U)
    # Convert to matrices
    U_mat = [U[i] for i in range(n*m)] # Wait, U is list of rows?
    # The interface expects U to be (n*m, rank).
    # My U is a list of rank vectors, each of length n*m.
    # So U is rank x n*m.
    # I need to transpose.
    
    U_final = [[U[c][i] for c in range(rank)] for i in range(n*m)]
    V_final = [[V[c][j] for c in range(rank)] for j in range(m*p)]
    W_final = [[W[c][k] for c in range(rank)] for k in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
