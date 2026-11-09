"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrix multiplication (Rank 7)
    # Indices: i, j in 0..1 for A and B. k in 0..1 for C.
    # U is 4x7, V is 4x7, W is 4x7.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_2x2 = [
        [1, 1, 0, 0, 0, 0, 0], # 00
        [0, 0, 1, 0, 1, -1, 1], # 01
        [0, 1, 0, 1, 0, 1, 0], # 10
        [0, 0, 0, 0, 1, 0, -1]  # 11
    ]
    # Wait, Strassen coefficients need to be checked.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U (for A)
    # Col 0 (M1): A00=1, A11=1
    # Col 1 (M2): A10=1, A11=1
    # Col 2 (M3): A00=1
    # Col 3 (M4): A11=1
    # Col 4 (M5): A00=1, A01=1
    # Col 5 (M6): A10=1, A00=-1
    # Col 6 (M7): A01=1, A11=-1
    
    U_2x2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    # V (for B)
    # Col 0 (M1): B00=1, B11=1
    # Col 1 (M2): B00=1
    # Col 2 (M3): B10=1, B11=-1
    # Col 3 (M4): B01=1, B00=-1
    # Col 4 (M5): B11=1
    # Col 5 (M6): B00=1, B01=1
    # Col 6 (M7): B10=1, B11=1
    
    V_2x2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    # W (for C)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Schoolbook decomposition for 3x3 matrix multiplication (Rank 27)
    U_3x3 = [[0]*27 for _ in range(9)]
    V_3x3 = [[0]*27 for _ in range(9)]
    W_3x3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U_3x3[i*3+k][col] = 1
                V_3x3[k*3+j][col] = 1
                W_3x3[i*3+j][col] = 1
                col += 1
                
    rank_2x2 = 7
    rank_3x3 = 27
    total_rank = rank_2x2 * rank_3x3
    
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    for m_idx in range(rank_3x3):
        for n_idx in range(rank_2x2):
            r_tot = m_idx * rank_2x2 + n_idx
            
            # Iterate over all 3x3 block indices and 2x2 internal indices
            for I in range(3):
                for J in range(3):
                    for K in range(3):
                        for r in range(2):
                            for s in range(2):
                                for t in range(2):
                                    # U index: A[I, K] block, internal [r, t]
                                    # i = 2*I + r, k_block = K, t_internal = t
                                    # Wait, U is for A. A indices are (i, k).
                                    # i = 2*I + r. k = 2*K + t.
                                    u_val = U_3x3[I*3+K][m_idx] * U_2x2[r*2+t][n_idx]
                                    U[(2*I+r)*6 + (2*K+t)][r_tot] += u_val
                                    
                                    # V index: B[K, J] block, internal [t, s]
                                    # k = 2*K + t, j = 2*J + s
                                    v_val = V_3x3[K*3+J][m_idx] * V_2x2[t*2+s][n_idx]
                                    V[(2*K+t)*6 + (2*J+s)][r_tot] += v_val
                                    
                                    # W index: C[I, J] block, internal [r, s]
                                    # i = 2*I + r, j = 2*J + s
                                    w_val = W_3x3[I*3+J][m_idx] * W_2x2[r*2+s][n_idx]
                                    W[(2*I+r)*6 + (2*J+s)][r_tot] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
