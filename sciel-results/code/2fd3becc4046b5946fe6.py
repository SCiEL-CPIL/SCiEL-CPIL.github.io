import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 49 (0-3) + 49 (4-7) + 16 (8) = 114
    total_rank = 114
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    # Terms 0..6
    # M1: A00+A11, B00+B11 -> C00+=1, C11+=1
    # M2: A10+A11, B00     -> C10+=1
    # M3: A00,     B10-B11 -> C01+=1, C11+=1
    # M4: A11,     B01-B00 -> C00-=1, C10+=1
    # M5: A00+A01, B11     -> C00+=1
    # M6: A10-A00, B00+B01 -> C11+=1
    # M7: A01-A11, B10+B11 -> C01+=1, C11-=1
    
    # Inner Strassen definitions (for 2x2 blocks)
    # cA[t][I][J], cB[t][J][K], cW[t][K][I]
    # I,J,K in 0,1
    
    cA = [
        [[1,0],[0,1]], # M1: A00 + A11
        [[0,0],[1,1]], # M2: A10 + A11
        [[1,0],[0,0]], # M3: A00
        [[0,0],[0,1]], # M4: A11
        [[1,1],[0,0]], # M5: A00 + A01
        [[-1,0],[1,0]],# M6: A10 - A00
        [[0,-1],[0,1]] # M7: A01 - A11
    ]
    
    cB = [
        [[1,0],[0,1]], # M1: B00 + B11
        [[1,0],[0,0]], # M2: B00
        [[0,1],[0,-1]],# M3: B10 - B11
        [[-1,1],[0,0]],# M4: B01 - B00
        [[0,0],[0,1]], # M5: B11
        [[1,1],[0,0]], # M6: B00 + B01
        [[0,1],[0,1]]  # M7: B10 + B11
    ]
    
    cW = [
        [[1,0],[0,1]], # M1: C00 +1, C11 +1
        [[0,0],[1,0]], # M2: C10 +1
        [[0,1],[0,1]], # M3: C01 +1, C11 +1
        [[-1,0],[1,0]],# M4: C00 -1, C10 +1
        [[1,0],[0,0]], # M5: C00 +1
        [[0,0],[0,1]], # M6: C11 +1
        [[0,1],[0,-1]] # M7: C01 +1, C11 -1
    ]

    # Outer Strassen for 4x4x4
    # Outer terms T1..T7
    # A_outer[t] is combo of 2x2 blocks (I,J)
    # B_outer[t] is combo of 2x2 blocks (J,K)
    # W_outer[t] is combo of 2x2 blocks (K,I)
    
    # Note: The "inner" Strassen is applied to the result of the "outer" linear combos?
    # No, the outer Strassen combines the *blocks*.
    # A_t = sum I,J cOuterA[t][I][J] A[I][J]
    # Then M_t = A_t B_t is a 2x2 matrix multiplication of the resulting 2x2 matrices?
    # No. A_t is a 2x2 matrix formed by the blocks.
    # So M_t is the product of two 2x2 matrices.
    # So we use the Inner Strassen decomposition for the multiplication of these 2x2 matrices.
    # The indices for U, V, W in the inner step correspond to the *elements* of these 2x2 matrices.
    # Which correspond to the *blocks* of the 4x4 matrices.
    
    # Let's refine the mapping.
    # Global U index: i_g * 4 + j_g
    # i_g = I * 2 + i_in
    # j_g = J * 2 + j_in
    # Inner U index for block (I,J) and element (i_in, j_in) is just the element.
    # But the Inner Strassen U values are for the 2x2 tensor.
    # The 2x2 tensor indices are (i_in, j_in, k_in).
    # U_val = cA_inner[t_in][i_in][j_in]? No.
    # In the 2x2 Strassen, U[i*2+j, term] is the coefficient of A[i,j].
    # So for the 4x4 case, for a fixed Outer term t_out and Inner term t_in:
    # The coefficient for A[I,J] element (i_in, j_in) is:
    # cOuterA[t_out][I][J] * cA_inner[t_in][i_in][j_in]?
    # Wait, cA_inner is defined for the 2x2 blocks.
    # In the inner step, the "matrix" is the 2x2 block.
    # So A_inner[i_in, j_in] is the element (i_in, j_in) of the block A[I,J].
    # So yes, U_val = cOuterA[t_out][I][J] * cA_inner[t_in][i_in][j_in].
    
    # We need cA_inner, cB_inner, cW_inner for the 2x2 Strassen.
    # These are the same as cA, cB, cW defined above?
    # Yes, cA[t][i][j] is the coefficient of A[i,j] in M_t.
    
    cOuterA = [
        [[1,0],[0,1]], # T1: A00 + A11
        [[0,0],[1,1]], # T2: A10 + A11
        [[1,0],[0,0]], # T3: A00
        [[0,0],[0,1]], # T4: A11
        [[1,1],[0,0]], # T5: A00 + A01
        [[-1,0],[1,0]],# T6: A10 - A00
        [[0,-1],[0,1]] # T7: A01 - A11
    ]
    
    cOuterB = [
        [[1,0],[0,1]], # T1: B00 + B11
        [[1,0],[0,0]], # T2: B00
        [[0,1],[0,-1]],# T3: B10 - B11
        [[-1,1],[0,0]],# T4: B01 - B00
        [[0,0],[0,1]], # T5: B11
        [[1,1],[0,0]], # T6: B00 + B01
        [[0,1],[0,1]]  # T7: B10 + B11
    ]
    
    cOuterW = [
        [[1,0],[0,1]], # T1: C00 +1, C11 +1
        [[0,0],[1,0]], # T2: C10 +1
        [[0,1],[0,1]], # T3: C01 +1, C11 +1
        [[-1,0],[1,0]],# T4: C00 -1, C10 +1
        [[1,0],[0,0]], # T5: C00 +1
        [[0,0],[0,1]], # T6: C11 +1
        [[0,1],[0,-1]] # T7: C01 +1, C11 -1
    ]

    def add_strassen_block(b_offset, rank_offset):
        nonlocal col
        for t_out in range(7):
            for t_in in range(7):
                # Determine coefficients for U, V, W
                # U depends on I, J, i_in, j_in
                # V depends on J, K, j_in, k_in
                # W depends on K, I, k_in, i_in
                
                # cU[i_in][j_in] for each I,J is cOuterA[t_out][I][J] * cA[t_in][i_in][j_in]
                # cV[j_in][k_in] for each J,K is cOuterB[t_out][J][K] * cB[t_in][j_in][k_in]
                # cW[k_in][i_in] for each K,I is cOuterW[t_out][K][I] * cW[t_in][k_in][i_in]
                
                for I in range(2):
                    for J in range(2):
                        cA_block = cOuterA[t_out][I][J]
                        if cA_block == 0: continue
                        for i_in in range(2):
                            for j_in in range(2):
                                c_val = cA_block * cA[t_in][i_in][j_in]
                                if c_val == 0: continue
                                i_g = I * 2 + i_in
                                j_g = J * 2 + j_in
                                u_idx = i_g * m + j_g
                                U[u_idx][col] += c_val
                                
                for J in range(2):
                    for K in range(2):
                        cB_block = cOuterB[t_out][J][K]
                        if cB_block == 0: continue
                        for j_in in range(2):
                            for k_in in range(2):
                                c_val = cB_block * cB[t_in][j_in][k_in]
                                if c_val == 0: continue
                                j_g = J * 2 + j_in
                                # k_g is local to the 4x4 block, offset by b_offset
                                k_g = b_offset + K * 2 + k_in
                                v_idx = j_g * p + k_g
                                V[v_idx][col] += c_val
                                
                for K in range(2):
                    for I in range(2):
                        cW_block = cOuterW[t_out][K][I]
                        if cW_block == 0: continue
                        for k_in in range(2):
                            for i_in in range(2):
                                c_val = cW_block * cW[t_in][k_in][i_in]
                                if c_val == 0: continue
                                k_g = b_offset + K * 2 + k_in
                                i_g = I * 2 + i_in
                                w_idx = k_g * n + i_g
                                W[w_idx][col] += c_val
                                
                col += 1

    # Add first 4 columns of B (b_offset 0)
    add_strassen_block(0, 0)
    # Add next 4 columns of B (b_offset 4)
    add_strassen_block(4, 49)
    
    # Add last column (b_offset 8), schoolbook rank 16
    # B is 4x1.
    # C_i = sum_j A_ij B_j0
    # U[i*m+j, col] = 1
    # V[j*p+8, col] = 1
    # W[8*n+i, col] = 1
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+8][col] = 1
            W[8*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
