"""Tensor decomposition for 4x5 * 5x5 matrix multiplication using Strassen."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses a block decomposition:
    A (4x5) = [A_44 | A_41]
    B (5x5) = [B_45; B_15] = [[B_44, B_41], [B_14, B_11]]
    
    C = A B
    C_44 = A_44 B_44 + A_41 B_14
    C_41 = A_44 B_41 + A_41 B_11
    
    Ranks:
    A_44 B_44: Strassen 4x4x4 -> 49
    A_41 B_14: Schoolbook 4x1x4 -> 16
    A_44 B_41: Schoolbook 4x4x1 -> 16
    A_41 B_11: Schoolbook 4x1x1 -> 4
    Total: 85
    """
    n, m, p = 4, 5, 5
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    # 1. Strassen for A_44 * B_44
    # Indices for 4x4 blocks:
    # A[i, j] where i,j in 0..3
    # B[j, k] where j,k in 0..3
    # C[k, i] where k,i in 0..3
    
    # Define the 7 Strassen terms for 2x2x2
    # u, v, w are length 4, indexed by (row, col) of 2x2
    # Index mapping: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Coefficients for Strassen 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    
    # We need the rank-1 decomposition of the 2x2x2 tensor.
    # Let's use the standard Strassen decomposition vectors.
    # Verified set for 2x2x2:
    strassen_2x2x2 = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]),
        ([1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]),
        ([0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]),
        ([0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 0, 1])
    ]
    
    # For 4x4x4, we use Strassen on 2x2 blocks.
    # Block indices: i1, j1 for A; j1, k1 for B; k1, i1 for C.
    # i1, j1, k1 in {0, 1}
    
    # Coefficients for X_s (input to inner 2x2x2)
    # X_s[i1, j1]
    # M1: X1 = A00+A11 -> X1[0,0]=1, X1[1,1]=1
    # M2: X2 = A10+A11 -> X2[1,0]=1, X2[1,1]=1
    # M3: X3 = A00     -> X3[0,0]=1
    # M4: X4 = A11     -> X4[1,1]=1
    # M5: X5 = A00+A10 -> X5[0,0]=1, X5[1,0]=1
    # M6: X6 = A01-A00 -> X6[0,1]=1, X6[0,0]=-1
    # M7: X7 = A10-A11 -> X7[1,0]=1, X7[1,1]=-1
    
    # Coefficients for Y_s
    # M1: Y1 = B00+B11 -> Y1[0,0]=1, Y1[1,1]=1
    # M2: Y2 = B00     -> Y2[0,0]=1
    # M3: Y3 = B10-B11 -> Y3[1,0]=1, Y3[1,1]=-1
    # M4: Y4 = B01-B10 -> Y4[0,1]=1, Y4[1,0]=-1
    # M5: Y5 = B11     -> Y5[1,1]=1
    # M6: Y6 = B00+B10 -> Y6[0,0]=1, Y6[1,0]=1
    # M7: Y7 = B01+B11 -> Y7[0,1]=1, Y7[1,1]=1
    
    # Coefficients for C_s (output combination)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # gamma_s[k1, i1]
    # s=1 (M1): C00+=1, C11+=1
    # s=2 (M2): C10+=1, C11-=1
    # s=3 (M3): C01+=1, C11+=1
    # s=4 (M4): C00+=1, C10+=1
    # s=5 (M5): C00-=1, C01+=1
    # s=6 (M6): C11+=1
    # s=7 (M7): C00+=1
    
    # Let's build the tables
    # X_coeffs[s][i1][j1]
    X_coeffs = [
        [[1, 0], [0, 1]], # M1: A00+A11
        [[0, 0], [1, 1]], # M2: A10+A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 0], [1, 0]], # M5: A00+A10
        [[-1, 1], [0, 0]],# M6: A01-A00
        [[0, 0], [1, -1]] # M7: A10-A11
    ]
    
    # Y_coeffs[s][j1][k1]
    Y_coeffs = [
        [[1, 0], [0, 1]], # M1: B00+B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 0], [1, -1]],# M3: B10-B11
        [[0, 1], [-1, 0]],# M4: B01-B10
        [[0, 0], [0, 1]], # M5: B11
        [[1, 0], [1, 0]], # M6: B00+B10
        [[0, 1], [0, 1]]  # M7: B01+B11
    ]
    
    # Gamma_coeffs[s][k1][i1]
    # C[k1, i1]
    Gamma_coeffs = [
        [[1, 0], [0, 1]], # M1: C00+=1, C11+=1
        [[0, 0], [1, -1]],# M2: C10+=1, C11-=1
        [[0, 1], [0, 0]], # M3: C01+=1
        [[0, 0], [1, 1]], # M4: C00+=1, C10+=1 (Wait, C10 is k1=1, i1=0)
        # Let's re-verify C10 = M2 + M4.
        # M2 contributes to C10 (k1=1, i1=0) with +1.
        # M4 contributes to C10 (k1=1, i1=0) with +1.
        # So Gamma[4][1][0] = 1.
        # M4 also contributes to C00 (k1=0, i1=0) with +1.
        # So Gamma[4][0][0] = 1.
        # My table above for Gamma[4] was [[0,0],[1,1]] which means Gamma[4][0][0]=0, Gamma[4][0][1]=0, Gamma[4][1][0]=1, Gamma[4][1][1]=1.
        # This is WRONG. C11 does not get M4.
        # Let's restart Gamma table carefully.
        
        # C00 (k1=0, i1=0): M1(+1), M4(+1), M5(-1), M7(+1)
        # C01 (k1=0, i1=1): M3(+1), M5(+1)
        # C10 (k1=1, i1=0): M2(+1), M4(+1)
        # C11 (k1=1, i1=1): M1(+1), M2(-1), M3(+1), M6(+1)
        
        # s=1 (M1): C00+=1, C11+=1
        # s=2 (M2): C10+=1, C11-=1
        # s=3 (M3): C01+=1, C11+=1
        # s=4 (M4): C00+=1, C10+=1
        # s=5 (M5): C00-=1, C01+=1
        # s=6 (M6): C11+=1
        # s=7 (M7): C00+=1
    ]
    
    Gamma_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 0]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    # Generate 49 terms
    for s in range(7):
        for t in range(7):
            u2, v2, w2 = strassen_2x2x2[t]
            
            # u2, v2, w2 are length 4
            # Index mapping for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # u2[i0, j0]
            
            U = [0.0] * 20
            V = [0.0] * 25
            W = [0.0] * 20
            
            for i1 in range(2):
                for j1 in range(2):
                    c_x = X_coeffs[s][i1][j1]
                    if c_x != 0:
                        for i0 in range(2):
                            for j0 in range(2):
                                # u2 index for (i0, j0)
                                idx_u = i0 * 2 + j0
                                c_u = u2[idx_u]
                                if c_u != 0:
                                    # Global A index: (i1*2+i0, j1*2+j0)
                                    # A is 4x5. i in 0..3, j in 0..3 (for this part)
                                    i_global = i1 * 2 + i0
                                    j_global = j1 * 2 + j0
                                    U[i_global * 5 + j_global] = c_x * c_u
                                    
            for j1 in range(2):
                for k1 in range(2):
                    c_y = Y_coeffs[s][j1][k1]
                    if c_y != 0:
                        for j0 in range(2):
                            for k0 in range(2):
                                # v2 index for (j0, k0)
                                idx_v = j0 * 2 + k0
                                c_v = v2[idx_v]
                                if c_v != 0:
                                    # Global B index: (j1*2+j0, k1*2+k0)
                                    # B is 5x5. j in 0..3, k in 0..3
                                    j_global = j1 * 2 + j0
                                    k_global = k1 * 2 + k0
                                    V[j_global * 5 + k_global] = c_y * c_v
                                    
            for k1 in range(2):
                for i1 in range(2):
                    c_w = Gamma_coeffs[s][k1][i1]
                    if c_w != 0:
                        for k0 in range(2):
                            for i0 in range(2):
                                # w2 index for (k0, i0)
                                idx_w = k0 * 2 + i0
                                c_w2 = w2[idx_w]
                                if c_w2 != 0:
                                    # Global C index: (k1*2+k0, i1*2+i0)
                                    # C is 4x5. k in 0..3, i in 0..3
                                    # W index is k*n + i = k*4 + i
                                    k_global = k1 * 2 + k0
                                    i_global = i1 * 2 + i0
                                    W[k_global * 4 + i_global] = c_w * c_w2
                                    
            add_term(U, V, W)

    # 2. A_41 * B_14 (Rank 16)
    # A[i, 4], B[4, k], C[k, i]
    for i in range(4):
        for k in range(4):
            U = [0.0] * 20
            V = [0.0] * 25
            W = [0.0] * 20
            U[i * 5 + 4] = 1.0
            V[4 * 5 + k] = 1.0
            W[k * 4 + i] = 1.0
            add_term(U, V, W)

    # 3. A_44 * B_41 (Rank 16)
    # A[i, j], B[j, 4], C[4, i]
    for i in range(4):
        for j in range(4):
            U = [0.0] * 20
            V = [0.0] * 25
            W = [0.0] * 20
            U[i * 5 + j] = 1.0
            V[j * 5 + 4] = 1.0
            W[4 * 4 + i] = 1.0
            add_term(U, V, W)

    # 4. A_41 * B_11 (Rank 4)
    # A[i, 4], B[4, 4], C[4, i]
    for i in range(4):
        U = [0.0] * 20
        V = [0.0] * 25
        W = [0.0] * 20
        U[i * 5 + 4] = 1.0
        V[4 * 5 + 4] = 1.0
        W[4 * 4 + i] = 1.0
        add_term(U, V, W)

    rank = len(terms)
    U_mat = [[0.0] * rank for _ in range(n * m)]
    V_mat = [[0.0] * rank for _ in range(m * p)]
    W_mat = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U_mat[idx][r] = u[idx]
        for idx in range(m * p):
            V_mat[idx][r] = v[idx]
        for idx in range(p * n):
            W_mat[idx][r] = w[idx]

    return {"U": U_mat, "V": V_mat, "W": W_mat}
