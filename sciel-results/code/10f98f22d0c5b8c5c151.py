import numpy as np

def construct():
    n, m, p = 2, 5, 5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off):
        nonlocal col
        # Strassen indices mapping
        # A indices: (a_i_off + di, a_j_off + dj)
        # B indices: (b_j_off + dj, b_k_off + dk)
        # C indices: (c_i_off + di, c_k_off + dk)
        
        # P1 = A00 * (B00 - B10)
        # P2 = (A00 + A01) * B10
        # P3 = (A01 - A11) * B11
        # P4 = A11 * (B01 - B00)
        # P5 = (A00 + A11) * (B00 + B11)
        # P6 = (A01 + A11) * (B00 - B01)
        # P7 = (A00 + A01) * (B10 + B11)
        
        # C00 = P5 + P4 - P3 + P7
        # C01 = P1 + P5
        # C10 = P2 + P3
        # C11 = P5 + P1 - P2 + P6
        
        # We need to map these to U, V, W columns.
        # U row: a_i_off + di, a_j_off + dj
        # V row: b_j_off + dj, b_k_off + dk
        # W row: c_k_off + dk, c_i_off + di
        
        # P1
        U[a_i_off + 0][col] += 1
        U[a_i_off + 1][col] += 0 # A00 is at (0,0)
        # Wait, A00 is at (a_i_off, a_j_off).
        # A indices in U are flattened: i*m + j.
        
        # Let's define indices clearly.
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        # P1: A00 * B00 - A00 * B10
        # A00: i=0, j=0
        # B00: j=0, k=0
        # B10: j=1, k=0
        
        # We add terms to U, V, W.
        
        # P1
        U[a_i_off + 0][col] += 1
        V[b_j_off + 0][col] += 1
        V[b_j_off + 1][col] -= 1
        W[c_k_off + 0][col] += 1
        W[c_k_off + 1][col] += 0 # C01? No.
        # C00 gets P1? No.
        # C01 = P1 + P5.
        # C00 = P5 + P4 - P3 + P7.
        
        # Let's just fill the columns for each P.
        
        # P1
        U[a_i_off + 0][col] += 1
        V[b_j_off + 0][col] += 1
        V[b_j_off + 1][col] -= 1
        W[c_k_off + 1][col] += 1 # C01
        col += 1
        
        # P2
        U[a_i_off + 0][col] += 1
        U[a_i_off + 0][col] += 1 # A00+A01? No.
        # P2 = (A00 + A01) * B10
        # A00: (0,0), A01: (0,1)
        # B10: (1,0)
        # C10 = P2 + P3.
        U[a_i_off + 0][col] += 1
        U[a_i_off + 0][col] += 1 # This is wrong. A00 and A01 are different rows in U?
        # A00 is i=0, j=0. A01 is i=0, j=1.
        # U index for A00: 0*m + 0 = 0.
        # U index for A01: 0*m + 1 = 1.
        # So U[0][col] += 1, U[1][col] += 1.
        # Reset and do properly.
        
        # This manual filling is error prone.
        pass

    # Better approach: Generate the full Strassen U, V, W for 2x2x2 and then map.
    
    # Strassen 2x2x2 U, V, W
    # U (4x7), V (4x7), W (4x7)
    # Indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Note: W index is k*n+i.
    # W for C00 (i=0,k=0) is 0*2+0=0.
    # W for C01 (i=0,k=1) is 1*2+0=2.
    # W for C10 (i=1,k=0) is 0*2+1=1.
    # W for C11 (i=1,k=1) is 1*2+1=3.
    
    # Let's hardcode the Strassen U, V, W for 2x2x2.
    
    # P1: A00(B00-B10) -> C01
    # P2: (A00+A01)B10 -> C10
    # P3: (A01-A11)B11 -> C10
    # P4: A11(B01-B00) -> C00
    # P5: (A00+A11)(B00+B11) -> C00, C01, C11
    # P6: (A01+A11)(B00-B01) -> C11
    # P7: (A00+A01)(B10+B11) -> C00
    
    # This is getting complex to hardcode correctly in the fly.
    # I will write a function to generate the 2x2x2 Strassen decomposition.
    
    def get_strassen_2x2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # P1: A00 * (B00 - B10) -> C01
        u[0][0] = 1
        v[0][0] = 1
        v[2][0] = -1 # B10 is j=1, k=0 -> idx 1*2+0=2? No.
        # V index: j*p + k. p=2.
        # B00: j=0, k=0 -> 0
        # B10: j=1, k=0 -> 2
        # B01: j=0, k=1 -> 1
        # B11: j=1, k=1 -> 3
        
        # W index: k*n + i. n=2.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        
        # P1
        u[0][0] = 1
        v[0][0] = 1
        v[2][0] = -1
        w[2][0] = 1 # C01
        
        # P2: (A00 + A01) * B10 -> C10
        u[0][1] = 1
        u[1][1] = 1
        v[2][1] = 1
        w[1][1] = 1 # C10
        
        # P3: (A01 - A11) * B11 -> C10
        u[1][2] = 1
        u[3][2] = -1
        v[3][2] = 1
        w[1][2] = 1 # C10
        
        # P4: A11 * (B01 - B00) -> C00
        u[3][3] = 1
        v[1][3] = 1
        v[0][3] = -1
        w[0][3] = 1 # C00
        
        # P5: (A00 + A11) * (B00 + B11) -> C00, C01, C11
        u[0][4] = 1
        u[3][4] = 1
        v[0][4] = 1
        v[3][4] = 1
        w[0][4] = 1 # C00
        w[2][4] = 1 # C01
        w[3][4] = 1 # C11
        
        # P6: (A01 + A11) * (B00 - B01) -> C11
        u[1][5] = 1
        u[3][5] = 1
        v[0][5] = 1
        v[1][5] = -1
        w[3][5] = 1 # C11
        
        # P7: (A00 + A01) * (B10 + B11) -> C00
        u[0][6] = 1
        u[1][6] = 1
        v[2][6] = 1
        v[3][6] = 1
        w[0][6] = 1 # C00
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2x2()
    
    # Add block
    def add_block(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if s_u[i*2+j][r] != 0:
                        U[(a_i+i)*m + (a_j+j)][col] += s_u[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    if s_v[j*2+k][r] != 0:
                        V[(b_j+j)*p + (b_k+k)][col] += s_v[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    if s_w[k*2+i][r] != 0:
                        W[(c_k+k)*n + (c_i+i)][col] += s_w[k*2+i][r]
            col += 1

    # 1. A_L B_TL
    # A_L cols 0-3. B_TL rows 0-3, cols 0-3.
    # Blocks:
    # A_L0 (cols 0,1) * B_00 (rows 0,1; cols 0,1) -> C_L0 (cols 0,1)
    add_block(0, 0, 0, 0, 0, 0)
    # A_L1 (cols 2,3) * B_10 (rows 2,3; cols 0,1) -> C_L0 (cols 0,1)
    add_block(0, 2, 2, 0, 0, 0)
    # A_L0 (cols 0,1) * B_01 (rows 0,1; cols 2,3) -> C_L1 (cols 2,3)
    add_block(0, 0, 0, 2, 0, 2)
    # A_L1 (cols 2,3) * B_11 (rows 2,3; cols 2,3) -> C_L1 (cols 2,3)
    add_block(0, 2, 2, 2, 0, 2)
    
    # 2. A_R B_BL
    # A_R col 4. B_BL row 4, cols 0-3.
    # Schoolbook 8 terms.
    for k in range(4):
        for i in range(2):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # 3. A_L B_TR
    # A_L cols 0-3. B_TR rows 0-3, col 4.
    # Schoolbook 8 terms.
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
            
    # 4. A_R B_BR
    # A_R col 4. B_BR row 4, col 4.
    # Schoolbook 2 terms.
    for i in range(2):
        U[i*m + 4][col] = 1
        V[4*p + 4][col] = 1
        W[4*n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
