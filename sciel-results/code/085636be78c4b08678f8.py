"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # U2[r, m] where r is row of A (0,1), m is term (0..6)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    # V2[c, m] where c is col of A / row of B (0,1), m is term (0..6)
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    # W2[s, m] where s is col of B / row of C (0,1), m is term (0..6)
    # Note: In 2x2 Strassen, C[r, s] = sum_m W2[r, s, m] M_m?
    # My previous definition: W2[s, m] was for C[r, s]?
    # Let's re-verify W2 indexing.
    # C00 = M0 + M3 - M4 + M6. So for r=0, s=0: W2[0,0]=1, W2[3,0]=1, W2[4,0]=-1, W2[6,0]=1.
    # C01 = M2 + M4. So for r=0, s=1: W2[2,1]=1, W2[4,1]=1.
    # C10 = M1 + M3. So for r=1, s=0: W2[1,0]=1, W2[3,0]=1.
    # C11 = M0 - M1 + M2 + M5. So for r=1, s=1: W2[0,1]=1, W2[1,1]=-1, W2[2,1]=1, W2[5,1]=1.
    # My previous W2 was defined as rows corresponding to C[r, s]?
    # Let's define W2[r, s, m] or just build W directly.
    # I will use the delta coefficients directly.
    
    # beta[L, i, j]
    beta = [[0]*4 for _ in range(7)]
    beta[0][0] = 1; beta[0][3] = 1 # A00, A11 (i=0,j=0 -> 0; i=1,j=1 -> 3)
    beta[1][2] = 1; beta[1][3] = 1 # A10, A11 (i=1,j=0 -> 2; i=1,j=1 -> 3)
    beta[2][0] = 1                 # A00
    beta[3][3] = 1                 # A11
    beta[4][0] = 1; beta[4][1] = 1 # A00, A01
    beta[5][2] = 1; beta[5][0] = -1# A10, A00
    beta[6][1] = 1; beta[6][3] = -1# A01, A11

    # gamma[L, j, k]
    gamma = [[0]*4 for _ in range(7)]
    gamma[0][0] = 1; gamma[0][3] = 1 # B00, B11
    gamma[1][0] = 1                 # B00
    gamma[2][1] = 1; gamma[2][3] = -1# B01, B11
    gamma[3][2] = 1; gamma[3][0] = -1# B10, B00
    gamma[4][3] = 1                 # B11
    gamma[5][0] = 1; gamma[5][1] = 1# B00, B01
    gamma[6][2] = 1; gamma[6][3] = 1# B10, B11

    # delta[L, k, i]
    delta = [[0]*4 for _ in range(7)]
    # C00: M0, M3, -M4, M6
    delta[0][0] = 1
    delta[3][0] = 1
    delta[4][0] = -1
    delta[6][0] = 1
    # C01: M2, M4
    delta[2][1] = 1
    delta[4][1] = 1
    # C10: M1, M3
    delta[1][2] = 1
    delta[3][2] = 1
    # C11: M0, -M1, M2, M5
    delta[0][3] = 1
    delta[1][3] = -1
    delta[2][3] = 1
    delta[5][3] = 1

    # Build U, V, W
    # U[I, col] where I = 2*i + r
    # col = L * 7 + m
    for L in range(7):
        for m in range(7):
            col = L * 7 + m
            for i in range(2):
                for r in range(2):
                    I = 2 * i + r
                    u_val = U2[r][m] * beta[L][i*2 + (i*0)] # Wait, beta index is i, j.
                    # beta[L][i*2+j] is not correct. i is 0..1, j is 0..1.
                    # In my beta definition, I used i*2+j?
                    # i=0,j=0 -> 0. i=0,j=1 -> 1. i=1,j=0 -> 2. i=1,j=1 -> 3.
                    # So index is i*2+j.
                    # But U2 depends on r.
                    # U[I, col] = sum_j U2[r, m] * beta[L, i, j]
                    # Since U2[r, m] is independent of j, we can factor it out?
                    # No, beta[L, i, j] depends on j.
                    # U[I, col] = U2[r, m] * sum_j beta[L, i, j]
                    # Wait, is X_L[r, c] = sum_j beta[L, i, j] A_ij[r, c]?
                    # Yes.
                    # So for a fixed I (i, r), U[I, col] = U2[r, m] * (sum_j beta[L, i, j])?
                    # No. The term is U2[r, m] * A_ij[r, c].
                    # The coefficient of A_ij[r, c] in U is U2[r, m] * beta[L, i, j].
                    # But U is a matrix. U[I, col] is a single number.
                    # It must be that for a fixed I and col, the coefficient is constant.
                    # This implies that sum_j beta[L, i, j] must be the only factor?
                    # No. U[I, col] multiplies V[J, col].
                    # The product is sum_{I,J} U[I, col] V[J, col] A[I, :] B[:, J].
                    # This must equal sum_L sum_m ...
                    
                    # Let's check the tensor decomposition definition.
                    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
                    # Here i is (I, J)? No.
                    # U has rows indexed by (i, j) of A? No, U has rows indexed by input A indices.
                    # Input A is n x m = 4 x 4. Indices (i, j).
                    # Input B is m x p = 4 x 4. Indices (j, k).
                    # Output C is p x n = 4 x 4. Indices (k, i).
                    
                    # So U[i, j, r] is flattened to U[i*m + j, r].
                    # V[j, k, r] is flattened to V[j*p + k, r].
                    # W[k, i, r] is flattened to W[k*n + i, r].
                    
                    # My derivation:
                    # U[2i+r, col] = U2[r, m] * beta[L, i, j]?
                    # But U[2i+r, col] depends on j?
                    # No, the row index of U is 2i+r. This corresponds to A's row?
                    # A is 4x4. Row index is 0..3.
                    # My I = 2i+r. This is the row index of A.
                    # But A's entries are A[row, col].
                    # In the block structure, A[row, col] = A_{i, j}[r, c] where row=2i+r, col=2j+c.
                    # So U[row, col_factor] must be the coefficient of A[row, col] in the term.
                    # The term is U2[r, m] * X_L[r, c] * V2[c, m] * Y_L[c, s] ...
                    # X_L[r, c] = sum_j beta[L, i, j] A_{ij}[r, c] = sum_j beta[L, i, j] A[2i+r, 2j+c].
                    # So the coefficient of A[2i+r, 2j+c] is U2[r, m] * beta[L, i, j].
                    # This coefficient must be U[2i+r, col] * (something depending on j)?
                    # No, U[2i+r, col] is fixed for a given row and column factor.
                    # But the coefficient depends on j (the column of A).
                    # This means U cannot be separated from the column index j?
                    # Wait. The rank decomposition is T = sum_r u_r v_r w_r.
                    # u_r is a vector of length 16 (A's entries).
                    # v_r is a vector of length 16 (B's entries).
                    # w_r is a vector of length 16 (C's entries).
                    
                    # If the coefficient of A[2i+r, 2j+c] is U2[r, m] * beta[L, i, j],
                    # then u_r[2i+r, 2j+c] = U2[r, m] * beta[L, i, j].
                    # This is valid. u_r is just a 4x4 matrix flattened.
                    
                    # So:
                    # U[2i+r, col] is not enough. We need U[2i+r, 2j+c, col].
                    # But the interface requires U to be (n*m, rank) = (16, rank).
                    # Row index is 0..15.
                    # My I = 2i+r is only the row part.
                    # The full index is I_full = (2i+r)*4 + (2j+c).
                    # So U[(2i+r)*4 + (2j+c), col] = U2[r, m] * beta[L, i, j].
                    
                    # Similarly for V:
                    # V[(2j+c)*4 + (2k+s), col] = V2[c, m] * gamma[L, j, k].
                    
                    # Similarly for W:
                    # W[(2k+s)*4 + (2i+r), col] = W2[s, m] * delta[L, k, i].
                    # Wait, W2 indexing.
                    # C[2k+s, 2i+r] = C_{ki}[s, r].
                    # C_{ki}[s, r] = sum_L delta[L, k, i] M_L[s, r].
                    # M_L[s, r] = sum_m W2[s, r, m] P_{L, m}.
                    # So W[(2k+s)*4 + (2i+r), col] = W2[s, r, m] * delta[L, k, i].
                    
                    pass # Logic confirmed.

    # Implementation
    for L in range(7):
        for m in range(7):
            col = L * 7 + m
            for i in range(2):
                for r in range(2):
                    for j in range(2):
                        for c in range(2):
                            I_idx = (2 * i + r) * 4 + (2 * j + c)
                            U[I_idx][col] = U2[r][m] * beta[L][i * 2 + j]
                            
                            J_idx = (2 * j + c) * 4 + (2 * k + s) # Wait, k,s loop needed for V
                            
    # Correct loops for V and W
    for L in range(7):
        for m in range(7):
            col = L * 7 + m
            # U
            for i in range(2):
                for r in range(2):
                    for j in range(2):
                        for c in range(2):
                            I_idx = (2 * i + r) * 4 + (2 * j + c)
                            U[I_idx][col] = U2[r][m] * beta[L][i * 2 + j]
                            
            # V
            for j in range(2):
                for c in range(2):
                    for k in range(2):
                        for s in range(2):
                            J_idx = (2 * j + c) * 4 + (2 * k + s)
                            V[J_idx][col] = V2[c][m] * gamma[L][j * 2 + k]
                            
            # W
            for k in range(2):
                for s in range(2):
                    for i in range(2):
                        for r in range(2):
                            K_idx = (2 * k + s) * 4 + (2 * i + r)
                            # W2[s, r, m]
                            # We need W2 defined as W2[s, r, m] or similar.
                            # From earlier:
                            # C00 (r=0,s=0): M0, M3, -M4, M6
                            # C01 (r=0,s=1): M2, M4
                            # C10 (r=1,s=0): M1, M3
                            # C11 (r=1,s=1): M0, -M1, M2, M5
                            # Note: C[r, s] in 2x2 corresponds to C_{ki}[s, r]?
                            # In my notation C_{ki}[s, r] is row s, col r of the block C_ki.
                            # The block C_ki is at row 2k+s, col 2i+r of the 4x4 matrix.
                            # So W index is (2k+s)*4 + (2i+r).
                            # The value is W2_block[s, r, m] * delta[L, k, i].
                            
                            # Let's define W2_block[s, r, m]
                            # s is row of C block (0,1)
                            # r is col of C block (0,1)
                            # m is Strassen term (0..6)
                            
                            pass

    # Define W2_block
    W2_block = [[[0]*7 for _ in range(2)] for _ in range(2)] # [s][r][m]
    # s=0, r=0 (C00): M0, M3, -M4, M6
    W2_block[0][0][0] = 1
    W2_block[0][0][3] = 1
    W2_block[0][0][4] = -1
    W2_block[0][0][6] = 1
    # s=0, r=1 (C01): M2, M4
    W2_block[0][1][2] = 1
    W2_block[0][1][4] = 1
    # s=1, r=0 (C10): M1, M3
    W2_block[1][0][1] = 1
    W2_block[1][0][3] = 1
    # s=1, r=1 (C11): M0, -M1, M2, M5
    W2_block[1][1][0] = 1
    W2_block[1][1][1] = -1
    W2_block[1][1][2] = 1
    W2_block[1][1][5] = 1

    # Re-run W loop
    for L in range(7):
        for m in range(7):
            col = L * 7 + m
            for k in range(2):
                for s in range(2):
                    for i in range(2):
                        for r in range(2):
                            K_idx = (2 * k + s) * 4 + (2 * i + r)
                            W[K_idx][col] = W2_block[s][r][m] * delta[L][k * 2 + i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
