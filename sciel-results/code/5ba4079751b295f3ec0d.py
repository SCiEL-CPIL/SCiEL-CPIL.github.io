import time

def _strassen_2x2_base():
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1: (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5: (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6: (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7: (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 5
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 1. Fill Strassen 4x4x4 (Rank 49)
    U2, V2, W2 = _strassen_2x2_base()
    
    # Block indices for 2x2 blocks of 2x2 matrices
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # Global indices for A[bi*2+li, bj*2+lj]
    
    # Define the 7 Strassen block multiplications
    # Each is (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs is a dict {block_idx: coeff}
    # block_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    ops = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),      # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),      # M3
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 0: 1}),      # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 3: 1}),      # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1, 3: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 3: 1}) # M7
    ]
    
    col_offset = 0
    for op in ops:
        a_coeffs, b_coeffs, c_coeffs = op
        # For each of the 7 inner Strassen terms
        for c in range(7):
            u2_col = [U2[i][c] for i in range(4)]
            v2_col = [V2[i][c] for i in range(4)]
            w2_col = [W2[i][c] for i in range(4)]
            
            # u2_col corresponds to a00, a01, a10, a11 of the block matrix
            # a00 is block (0,0), a01 is block (0,1), a10 is block (1,0), a11 is block (1,1)
            # Indices in u2_col: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            
            # Fill U
            for local_idx in range(4):
                coeff_u = u2_col[local_idx]
                if coeff_u == 0: continue
                # Determine which block this local index corresponds to
                # local_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                bi = local_idx // 2
                bj = local_idx % 2
                block_idx = bi * 2 + bj # 0, 1, 2, 3
                a_coeff = a_coeffs.get(block_idx, 0)
                if a_coeff == 0: continue
                total_coeff = coeff_u * a_coeff
                # Add to global U
                for li in range(2):
                    for lj in range(2):
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        U[gi * m + gj][col_offset] += total_coeff
                        
            # Fill V
            for local_idx in range(4):
                coeff_v = v2_col[local_idx]
                if coeff_v == 0: continue
                bi = local_idx // 2
                bj = local_idx % 2
                block_idx = bi * 2 + bj
                b_coeff = b_coeffs.get(block_idx, 0)
                if b_coeff == 0: continue
                total_coeff = coeff_v * b_coeff
                for li in range(2):
                    for lj in range(2):
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        V[gi * p + gj][col_offset] += total_coeff
                        
            # Fill W
            for local_idx in range(4):
                coeff_w = w2_col[local_idx]
                if coeff_w == 0: continue
                bi = local_idx // 2
                bj = local_idx % 2
                block_idx = bi * 2 + bj
                c_coeff = c_coeffs.get(block_idx, 0)
                if c_coeff == 0: continue
                total_coeff = coeff_w * c_coeff
                for li in range(2):
                    for lj in range(2):
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        # W index: k*n + i. Here k is col index in C, i is row index.
                        # In C block (bi, bj), row is bi*2+li, col is bj*2+lj.
                        # So k = bj*2+lj, i = bi*2+li.
                        W[gj * n + gi][col_offset] += total_coeff
                        
            col_offset += 1
            
    # 2. Fill Schoolbook 4x4x1 (Rank 16)
    # C_5 = A * B_5
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i * m + j][col_offset] = 1
            V[j * p + 4][col_offset] = 1
            W[4 * n + i][col_offset] = 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
