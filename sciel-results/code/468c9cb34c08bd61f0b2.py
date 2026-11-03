"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 97 decomposition:
    # 1. Strassen for 4x4x4 (rank 49) for k=0..3
    # 2. Schoolbook for 4x4x3 (rank 48) for k=4..6
    
    rank = 49 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Outer Strassen coefficients for 2x2x2 blocks
    # M_u = A_u B_u
    # A_u, B_u are 2x2 matrices of blocks
    # Outer Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # We need to define the linear combinations for A_u and B_u
    # A_u is a 2x2 matrix. A_u[i,j] is a linear combo of A blocks.
    # A blocks: A00 (rows 0,1 cols 0,1), A01 (rows 0,1 cols 2,3), A10 (rows 2,3 cols 0,1), A11 (rows 2,3 cols 2,3)
    # Note: My block notation A11 corresponds to top-left.
    # Let's use 0-based block indices:
    # Block (0,0) = A00, Block (0,1) = A01, Block (1,0) = A10, Block (1,1) = A11
    
    # For each u in 1..7, define A_u and B_u
    # A_u is a 2x2 matrix of coefficients for the 4 blocks.
    # A_u[i,j] is a 2x2 matrix of coefficients for the entries in the block.
    # Actually, A_u is a 2x2 matrix where each entry is a 2x2 matrix of coefficients?
    # No, A_u is a 2x2 matrix formed by linear combinations of the 2x2 blocks.
    # So A_u[i,j] is a 2x2 matrix of coefficients for the entries of the block at position (i,j) in the block grid?
    # No, the block grid is 2x2.
    # A_u is a 2x2 matrix.
    # A_u[0,0] is a linear combination of A00.
    # A_u[0,1] is a linear combination of A01.
    # etc.
    
    # Let's define the coefficients for A_u and B_u for each u.
    # A_u[i,j] will be a 2x2 matrix of coefficients for the entries of the block.
    # But the blocks are added/subtracted whole.
    # So A_u[i,j] is either 0, I, or -I?
    # No, the blocks are 2x2 matrices.
    # A_u = \begin{pmatrix} A_{u,00} & A_{u,01} \\ A_{u,10} & A_{u,11} \end{pmatrix}
    # Where A_{u,ij} is a 2x2 matrix.
    # But in Strassen, we add blocks.
    # So A_{u,ij} is either 0, I, or -I?
    # No, the blocks are A00, A01, etc.
    # A_u[0,0] is a linear combination of A00?
    # In standard Strassen, we add the whole blocks.
    # So A_u[0,0] is a multiple of A00?
    # Yes, the coefficients are 1 or -1 for the whole block.
    
    # So for each u, we have a 2x2 grid of coefficients for the 4 blocks.
    # coeff_A[u][block_i][block_j] is a scalar (1, -1, 0).
    # This scalar multiplies the entire 2x2 block.
    
    # Let's define coeff_A and coeff_B for u=0..6 (0-indexed).
    # u=0: M1 = (A00+A11)(B00+B11)
    # u=1: M2 = (A10+A11)B00
    # u=2: M3 = A00(B01-B11)
    # u=3: M4 = A11(B10-B00)
    # u=4: M5 = (A00+A01)B11
    # u=5: M6 = (A10-A00)(B00+B01)
    # u=6: M7 = (A01-A11)(B10+B11)
    
    # Block indices:
    # 0: (0,0) -> rows 0,1 cols 0,1
    # 1: (0,1) -> rows 0,1 cols 2,3
    # 2: (1,0) -> rows 2,3 cols 0,1
    # 3: (1,1) -> rows 2,3 cols 2,3
    
    coeff_A = [
        [[1, 0], [0, 1]],   # u=0: A00 + A11
        [[0, 0], [1, 1]],   # u=1: A10 + A11
        [[1, 0], [0, 0]],   # u=2: A00
        [[0, 0], [0, 1]],   # u=3: A11
        [[1, 1], [0, 0]],   # u=4: A00 + A01
        [[-1, 0], [1, 0]],  # u=5: A10 - A00
        [[0, 1], [0, -1]]   # u=6: A01 - A11
    ]
    
    coeff_B = [
        [[1, 0], [0, 1]],   # u=0: B00 + B11
        [[1, 0], [0, 0]],   # u=1: B00
        [[0, 1], [0, -1]],  # u=2: B01 - B11
        [[0, -1], [1, 0]],  # u=3: B10 - B00
        [[0, 0], [0, 1]],   # u=4: B11
        [[1, 1], [0, 0]],   # u=5: B00 + B01
        [[0, 1], [1, 1]]    # u=6: B10 + B11
    ]
    
    # Inner Strassen for 2x2x2
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # For each inner term v (0..6), we need the coefficients for the entries of A_u and B_u.
    # A_u is a 2x2 matrix.
    # A_u[0,0], A_u[0,1], A_u[1,0], A_u[1,1].
    # Each A_u[i,j] is a 2x2 matrix of coefficients for the block entries?
    # No, A_u[i,j] is a scalar multiple of the block A_block.
    # So the coefficient for a specific entry A_global is coeff_A[u][block_i][block_j] * coeff_inner_A[v][i][j].
    # Wait, coeff_inner_A[v][i][j] is the coefficient of A_u[i,j] in the v-th inner term.
    
    # Let's define inner_coeff_A and inner_coeff_B for v=0..6.
    # These are 2x2 matrices of coefficients for the entries of A_u and B_u.
    
    inner_coeff_A = [
        [[1, 0], [0, 1]],   # v=0: A00 + A11 (of A_u)
        [[0, 0], [1, 1]],   # v=1: A10 + A11
        [[1, 0], [0, 0]],   # v=2: A00
        [[0, 0], [0, 1]],   # v=3: A11
        [[1, 1], [0, 0]],   # v=4: A00 + A01
        [[-1, 0], [1, 0]],  # v=5: A10 - A00
        [[0, 1], [0, -1]]   # v=6: A01 - A11
    ]
    
    inner_coeff_B = [
        [[1, 0], [0, 1]],   # v=0: B00 + B11
        [[1, 0], [0, 0]],   # v=1: B00
        [[0, 1], [0, -1]],  # v=2: B01 - B11
        [[0, -1], [1, 0]],  # v=3: B10 - B00
        [[0, 0], [0, 1]],   # v=4: B11
        [[1, 1], [0, 0]],   # v=5: B00 + B01
        [[0, 1], [1, 1]]    # v=6: B10 + B11
    ]
    
    # Output coefficients for inner Strassen
    # c00 = m0 + m3 - m4 - m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m6
    
    # Note: v indices 0..6 correspond to m0..m6.
    # c00: m0(1), m1(0), m2(0), m3(1), m4(-1), m5(0), m6(-1)
    # c01: m0(0), m1(0), m2(1), m3(0), m4(1), m5(0), m6(0)
    # c10: m0(0), m1(1), m2(0), m3(1), m4(0), m5(0), m6(0)
    # c11: m0(1), m1(-1), m2(1), m3(0), m4(0), m5(0), m6(1)
    
    inner_output_coeff = [
        [1, 0, 0, 1, -1, 0, -1],  # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 0, 1]    # c11
    ]
    
    # Outer output coefficients
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    
    outer_output_coeff = [
        [1, 0, 0, 1, -1, 0, -1],  # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 0, 1]    # C11
    ]
    
    # Now construct U, V, W for the 49 terms.
    for u in range(7):
        for v in range(7):
            # Fill U
            for bi in range(2):
                for bj in range(2):
                    # Block index in A_u
                    # A_u[bi, bj] is a 2x2 matrix of coefficients
                    # The coefficient for the block is coeff_A[u][bi][bj]
                    coeff_block_A = coeff_A[u][bi][bj]
                    if coeff_block_A == 0:
                        continue
                    # The block corresponds to global rows 2*bi .. 2*bi+1, cols 2*bj .. 2*bj+1
                    for li in range(2):
                        for lj in range(2):
                            # Local index in A_u[bi,bj]
                            # The coefficient for A_u[bi,bj][li,lj] in the v-th inner term is inner_coeff_A[v][li][lj]
                            coeff_inner_A = inner_coeff_A[v][li][lj]
                            coeff_total = coeff_block_A * coeff_inner_A
                            if coeff_total != 0:
                                global_i = 2 * bi + li
                                global_j = 2 * bj + lj
                                U[global_i * m + global_j][col] = coeff_total
            
            # Fill V
            for bj in range(2):
                for bk in range(2):
                    # Block index in B_u
                    # B_u[bj, bk] is a 2x2 matrix of coefficients
                    coeff_block_B = coeff_B[u][bj][bk]
                    if coeff_block_B == 0:
                        continue
                    for lj in range(2):
                        for lk in range(2):
                            coeff_inner_B = inner_coeff_B[v][lj][lk]
                            coeff_total = coeff_block_B * coeff_inner_B
                            if coeff_total != 0:
                                global_j = 2 * bj + lj
                                global_k = 2 * bk + lk
                                V[global_j * p + global_k][col] = coeff_total
            
            # Fill W
            # The term m_{u,v} contributes to the output blocks C_ab
            # The coefficient is outer_output_coeff[ab][u] * inner_output_coeff[local_i, local_j][v]
            # Wait, inner_output_coeff is indexed by [output_pos][v]
            # output_pos 0: c00, 1: c01, 2: c10, 3: c11
            # These correspond to local indices (0,0), (0,1), (1,0), (1,1) in M_u.
            
            for ab_i in range(2):
                for ab_j in range(2):
                    # Block index in C
                    # C_ab_i, ab_j
                    coeff_outer = outer_output_coeff[ab_i * 2 + ab_j][u]
                    if coeff_outer == 0:
                        continue
                    # The local output position in M_u corresponding to block (ab_i, ab_j) is ab_i * 2 + ab_j?
                    # No.
                    # M_u is a 2x2 matrix.
                    # M_u[0,0] contributes to C00.
                    # M_u[0,1] contributes to C01.
                    # M_u[1,0] contributes to C10.
                    # M_u[1,1] contributes to C11.
                    # So the local index is ab_i * 2 + ab_j.
                    local_pos = ab_i * 2 + ab_j
                    coeff_inner = inner_output_coeff[local_pos][v]
                    coeff_total = coeff_outer * coeff_inner
                    if coeff_total == 0:
                        continue
                    # This coefficient applies to all 4 entries of the block C_ab_i, ab_j
                    for li in range(2):
                        for lk in range(2):
                            global_i = 2 * ab_i + li
                            global_k = 2 * ab_j + lk
                            W[global_k * n + global_i][col] = coeff_total
            
            col += 1
    
    # --- Part 2: Schoolbook for 4x4x3 ---
    # k = 4, 5, 6
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
