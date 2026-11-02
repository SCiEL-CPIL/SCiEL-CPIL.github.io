import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Block 1: k=0..3 (4x4x4 Winograd)
    # Block 2: k=4..7 (4x4x4 Winograd)
    # Block 3: k=8 (4x4x1 Schoolbook)
    
    # Generate 4x4x4 Winograd decomposition
    # Based on Kronecker product of 2x2x2 Winograd (Rank 7)
    
    # 2x2x2 Winograd Matrices (Rank 7)
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, -1, 0, 0, 0, 0, 1],
        [0, 0, 1, -1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0]
    ], dtype=float)
    
    W2 = np.array([
        [0.5, 0, 0, 0, 0, 0, 0.5],
        [0, 0.5, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 0, 0.5, -0.5]
    ], dtype=float)
    
    # Verify 2x2 decomposition: C = W2 * (U2 A * V2 B)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U2 A = [a+c, a-c, b+d, b-d] ... wait, U2 is 4x7.
    # Let's check the first row of U2: [1, 1, 0, 0, 1, 0, 0].
    # This multiplies [a, b, c, d] to give a + c.
    # The 7 products are:
    # p1 = (a+c)(e+g)
    # p2 = (a+c)(f+h)
    # p3 = (a-c)(e+g) ... wait, U2 row 2 is [0,0,1,1,0,1,0] -> c+d?
    # My U2 above might be wrong.
    
    # Correct 2x2 Winograd U, V, W:
    # m1 = (a+c)(e+g)
    # m2 = (a+c)(f+h)
    # m3 = (b+d)(e+g)
    # m4 = (b+d)(f+h)
    # m5 = a(e-g)
    # m6 = b(f-h)
    # m7 = c(g-e) is not standard.
    
    # Let's use the standard F(2,2,2) decomposition from literature.
    # U = [[1, 1, 0, 0, 1, 0, 0],
    #      [0, 0, 1, 1, 0, 1, 0],
    #      [1, -1, 0, 0, 0, 0, 1],
    #      [0, 0, 1, -1, 0, 0, -1]]
    # This U maps [a,b,c,d] to:
    # u1 = a+c
    # u2 = c+d
    # u3 = a-c
    # u4 = c-d
    # u5 = a
    # u6 = b
    # u7 = c
    
    # V = [[1, 1, 1, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 1, 1, 1],
    #      [1, 1, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 1, 1, 0]]
    # This V maps [e,f,g,h] to:
    # v1 = e+g
    # v2 = f+h
    # v3 = e
    # v4 = g
    # v5 = f
    # v6 = h
    # v7 = f+h? No.
    
    # This is too risky to hardcode without verification.
    # I will use the schoolbook for 4x4x4 (Rank 64) for now?
    # Rank 64 + 64 + 16 = 144. No improvement.
    
    # I must get the Winograd decomposition right.
    # Let's use the property that M_4,4,4 = M_2,2,2 \otimes M_2,2,2.
    # If I have a correct rank-7 decomposition for 2x2, I can Kronecker it.
    
    # Correct Rank 7 decomposition for 2x2:
    # C00 = m1 + m5 + m7
    # C01 = m2 + m6
    # C10 = m3 + m7
    # C11 = m4 + m6
    # m1 = (a+c)(e+g)
    # m2 = (a+c)(f+h)
    # m3 = (b+d)(e+g)
    # m4 = (b+d)(f+h)
    # m5 = a(e-g)
    # m6 = b(f-h)
    # m7 = c(g-e)
    
    # Let's check m7: c(g-e) = cg - ce.
    # C10 = be + dg.
    # m3 + m7 = (b+d)(e+g) + c(g-e) = be + bg + de + dg + cg - ce.
    # This is not be + dg.
    
    # The correct 7th term is usually related to (a-d)(e-h) or similar.
    # Actually, the rank 7 decomposition is:
    # m1 = (a+c)(e+g)
    # m2 = (a+c)(f+h)
    # m3 = (b+d)(e+g)
    # m4 = (b+d)(f+h)
    # m5 = a(e-g)
    # m6 = b(f-h)
    # m7 = c(g-e) is WRONG.
    
    # Let's use the one from Winograd's paper.
    # m1 = (a+c)(e+g)
    # m2 = (a+c)(f+h)
    # m3 = (b+d)(e+g)
    # m4 = (b+d)(f+h)
    # m5 = a(e-g)
    # m6 = b(f-h)
    # m7 = (a-d)(e-h) ?
    
    # If I cannot guarantee the 2x2 decomposition, I will fall back to a safer but still improved method.
    # Is there a simpler improvement?
    # 4x4x9.
    # Split 9 into 4+4+1.
    # 4x4x4 Schoolbook is 64.
    # 4x4x4 Schoolbook is 64.
    # 4x4x1 Schoolbook is 16.
    # Total 144.
    
    # What if I split 4x4 into 2x2 blocks?
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12, B13], [B21, B22, B23]]
    # This doesn't reduce rank easily without Winograd.
    
    # I will implement the 2x2 Winograd decomposition and verify it in the code.
    # If it fails, I return the schoolbook.
    
    U2 = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, -1, 0, 0, 0, 0, 1],
        [0, 0, 1, -1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0]
    ])
    W2 = np.array([
        [0.5, 0, 0, 0, 0, 0, 0.5],
        [0, 0.5, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 0, 0.5, -0.5]
    ])
    
    # Verify
    A_test = np.array([[1, 2], [3, 4]], dtype=float)
    B_test = np.array([[5, 6], [7, 8]], dtype=float)
    C_test = A_test @ B_test
    
    U_A = U2 @ A_test.flatten()
    V_B = V2 @ B_test.flatten()
    P = U_A * V_B
    C_calc = (W2 @ P).reshape(2, 2)
    
    if not np.allclose(C_calc, C_test):
        # Fallback to schoolbook for the whole thing
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    # Note: The ordering of indices in Kronecker product must match the flattening order.
    # A is 4x4, flattened row-major: a00, a01, a02, a03, a10...
    # U2 maps [a00, a01, a10, a11] for 2x2.
    # For 4x4, we view it as 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The Kronecker product U2 \otimes U2 assumes the tensor structure.
    
    # Let's verify 4x4
    A4_test = np.random.rand(4, 4)
    B4_test = np.random.rand(4, 4)
    C4_test = A4_test @ B4_test
    
    U_A4 = U4 @ A4_test.flatten()
    V_B4 = V4 @ B4_test.flatten()
    P4 = U_A4 * V_B4
    C4_calc = (W4 @ P4).reshape(4, 4)
    
    if not np.allclose(C4_calc, C4_test):
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Construct full U, V, W
    rank_total = 49 + 49 + 16
    
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((36, rank_total))
    W_full = np.zeros((36, rank_total))
    
    # Block 1: k=0..3
    U_full[:, 0:49] = U4
    # V4 maps B1 (4x4) to 49 values.
    # B is 4x9. B1 is columns 0-3.
    # V_full rows 0..15 (j=0..3, k=0..3)
    V_full[0:16, 0:49] = V4
    # W4 maps product to C1 (4x4).
    # W_full rows 0..15 (k=0..3, i=0..3)
    W_full[0:16, 0:49] = W4
    
    # Block 2: k=4..7
    U_full[:, 49:98] = U4
    # V4 maps B2 (4x4) to 49 values.
    # B2 is columns 4-7.
    # V_full rows 16..31 (j=0..3, k=4..7)
    V_full[16:32, 49:98] = V4
    # W4 maps product to C2 (4x4).
    # W_full rows 16..31 (k=4..7, i=0..3)
    W_full[16:32, 49:98] = W4
    
    # Block 3: k=8
    # 4x4x1 Schoolbook
    U3 = np.eye(16)
    V3 = np.zeros((16, 16))
    # V3 maps B3 (4x1) to 16 values.
    # B3 is column 8.
    # V_full rows 32..35 (j=0..3, k=8)
    # V3 should be identity on the 4 entries of B3.
    # But V3 is 16x16? No, V3 is for the block.
    # The block rank is 16.
    # U3 is 16x16.
    # V3 is 16x16 (but only 4 rows of B are used).
    # W3 is 16x16.
    
    # U_full[:, 98:114] = U3
    U_full[:, 98:114] = U3
    
    # V3: Identity on the 4 entries of B3?
    # The product is sum_j A_ij B_j0.
    # We need 16 terms.
    # Term r corresponds to (i, j).
    # U3[i*4+j, r] = delta(i, r//4) delta(j, r%4)
    # V3[j*9+8, r] = delta(j, r%4)
    # W3[8*4+i, r] = delta(i, r//4)
    
    for r in range(16):
        i = r // 4
        j = r % 4
        U_full[i * 4 + j, 98 + r] = 1
        V_full[j * 9 + 8, 98 + r] = 1
        W_full[8 * 4 + i, 98 + r] = 1
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
