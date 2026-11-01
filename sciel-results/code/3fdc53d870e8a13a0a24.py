# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    
    # We'll use block decomposition:
    # Partition the 3x6 matrix into three 3x2 blocks
    # For each 3x2 block, use 2x3x2 decomposition with Strassen for 2x2x2 part
    # Total rank: 3 * 11 = 33
    
    rank = 33
    
    U = [[0.0] * rank for _ in range(n * m)]  # (6, 33)
    V = [[0.0] * rank for _ in range(m * p)]  # (18, 33)
    W = [[0.0] * rank for _ in range(p * n)]  # (12, 33)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Multiplying 2x2 by 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # For the tensor, U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, k] for input A element (i,j) at rank k
    # V[j*2+k, k] for input B element (j,k) at rank k  
    # W[k*2+i, k] for output C element (i,k) at rank k
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen, each shape (4, 7)"""
        U_s = [[0.0]*7 for _ in range(4)]
        V_s = [[0.0]*7 for _ in range(4)]
        W_s = [[0.0]*7 for _ in range(4)]
        
        # M1 = (a11+a22)(b11+b22)
        U_s[0][0] = 1.0  # a11
        U_s[3][0] = 1.0  # a22
        V_s[0][0] = 1.0  # b11
        V_s[3][0] = 1.0  # b22
        W_s[0][0] = 1.0  # c11
        W_s[3][0] = 1.0  # c22
        
        # M2 = (a21+a22)b11
        U_s[2][1] = 1.0  # a21
        U_s[3][1] = 1.0  # a22
        V_s[0][1] = 1.0  # b11
        W_s[2][1] = 1.0  # c21
        W_s[3][1] = -1.0 # c22
        
        # M3 = a11(b12-b22)
        U_s[0][2] = 1.0  # a11
        V_s[1][2] = 1.0  # b12
        V_s[3][2] = -1.0 # b22
        W_s[1][2] = 1.0  # c12
        W_s[3][2] = 1.0  # c22
        
        # M4 = a22(b21-b11)
        U_s[3][3] = 1.0  # a22
        V_s[2][3] = 1.0  # b21
        V_s[0][3] = -1.0 # b11
        W_s[0][3] = 1.0  # c11
        W_s[2][3] = 1.0  # c21
        
        # M5 = (a11+a12)b22
        U_s[0][4] = 1.0  # a11
        U_s[1][4] = 1.0  # a12
        V_s[3][4] = 1.0  # b22
        W_s[0][4] = -1.0 # c11
        W_s[1][4] = 1.0  # c12
        
        # M6 = (a21-a11)(b11+b12)
        U_s[2][5] = 1.0  # a21
        U_s[0][5] = -1.0 # a11
        V_s[0][5] = 1.0  # b11
        V_s[1][5] = 1.0  # b12
        W_s[3][5] = 1.0  # c22
        
        # M7 = (a12-a22)(b21+b22)
        U_s[1][6] = 1.0  # a12
        U_s[3][6] = -1.0 # a22
        V_s[2][6] = 1.0  # b21
        V_s[3][6] = 1.0  # b22
        W_s[0][6] = 1.0  # c11
        W_s[1][6] = 1.0  # c12
        
        return U_s, V_s, W_s
    
    def schoolbook_2x1x2():
        """Return (U, V, W) for 2x1x2 schoolbook, each shape (2, 4), (2, 4), (4, 4)"""
        U_s = [[0.0]*4 for _ in range(2)]
        V_s = [[0.0]*4 for _ in range(2)]
        W_s = [[0.0]*4 for _ in range(4)]
        
        # Output C[i,k] = A[i,0] * B[0,k]
        # For i in {0,1}, k in {0,1}:
        col = 0
        for i in range(2):
            for k in range(2):
                U_s[i][col] = 1.0
                V_s[k][col] = 1.0
                W_s[k*2 + i][col] = 1.0
                col += 1
        
        return U_s, V_s, W_s
    
    # For each of the 3 blocks (cols 0-1, 2-3, 4-5 of the 3x6 matrix):
    for block in range(3):
        col_offset = block * 11
        
        # Get Strassen 2x2x2
        U_str, V_str, W_str = strassen_2x2x2()
        # Get schoolbook 2x1x2
        U_sb, V_sb, W_sb = schoolbook_2x1x2()
        
        # Combine: the 2x3 matrix A is partitioned as [A_top (2x2) | A_bottom (2x1)]
        # Wait, I need to reconsider the partitioning.
        # 
        # A is 2x3: [[a00, a01, a02], [a10, a11, a12]]
        # B_block is 3x2: [[b00, b01], [b10, b11], [b20, b21]]
        #
        # Partition B into top 2x2 and bottom 1x2:
        # B_top = [[b00, b01], [b10, b11]] (rows 0,1)
        # B_bot = [[b20, b21]] (row 2)
        #
        # Partition A into left 2x2 and right 2x1:
        # A_left = [[a00, a01], [a10, a11]] (cols 0,1)
        # A_right = [[a02], [a12]] (col 2)
        #
        # C = A_left * B_top + A_right * B_bot
        #
        # A_left is 2x2, B_top is 2x2 → use Strassen (rank 7)
        # A_right is 2x1, B_bot is 1x2 → use schoolbook (rank 4)
        
        # Strassen part: A_left (2x2) * B_top (2x2)
        # A_left elements: a00, a01, a10, a11 → indices in U: 0,1,2,3 (for 2x2, row-major)
        # In the full 2x3 matrix A, these are at positions:
        # a00: row 0, col 0 → U index 0*3+0 = 0
        # a01: row 0, col 1 → U index 0*3+1 = 1
        # a10: row 1, col 0 → U index 1*3+0 = 3
        # a11: row 1, col 1 → U index 1*3+1 = 4
        #
        # B_top elements: b00, b01, b10, b11
        # In the full 3x6 matrix B, for block 'block':
        # b00: row 0, col block*2+0 → V index 0*6 + block*2+0
        # b01: row 0, col block*2+1 → V index 0*6 + block*2+1
        # b10: row 1, col block*2+0 → V index 1*6 + block*2+0
        # b11: row 1, col block*2+1 → V index 1*6 + block*2+1
        #
        # Output C elements for this block: 2x2 result
        # c00: row 0, col block*2+0 → W index (block*2+0)*2 + 0
        # c01: row 0, col block*2+1 → W index (block*2+1)*2 + 0
        # c10: row 1, col block*2+0 → W index (block*2+0)*2 + 1
        # c11: row 1, col block*2+1 → W index (block*2+1)*2 + 1
        
        # Map Strassen U indices (for 2x2 A_left) to full U indices
        # Strassen U index 0 = a11 → full U index 0
        # Strassen U index 1 = a12 → full U index 1
        # Strassen U index 2 = a21 → full U index 3
        # Strassen U index 3 = a22 → full U index 4
        
        strassen_to_full_U = [0, 1, 3, 4]  # a11, a12, a21, a22
        
        # Map Strassen V indices (for 2x2 B_top) to full V indices
        # Strassen V index 0 = b11 → full V index 0*6 + block*2+0
        # Strassen V index 1 = b12 → full V index 0*6 + block*2+1
        # Strassen V index 2 = b21 → full V index 1*6 + block*2+0
        # Strassen V index 3 = b22 → full V index 1*6 + block*2+1
        
        strassen_to_full_V = [0*6 + block*2+0, 0*6 + block*2+1, 1*6 + block*2+0, 1*6 + block*2+1]
        
        # Map Strassen W indices (for 2x2 C) to full W indices
        # Strassen W index 0 = c11 → full W index (block*2+0)*2 + 0
        # Strassen W index 1 = c12 → full W index (block*2+1)*2 + 0
        # Strassen W index 2 = c21 → full W index (block*2+0)*2 + 1
        # Strassen W index 3 = c22 → full W index (block*2+1)*2 + 1
        
        strassen_to_full_W = [(block*2+0)*2 + 0, (block*2+1)*2 + 0, (block*2+0)*2 + 1, (block*2+1)*2 + 1]
        
        # Add Strassen terms
        for r in range(7):
            for i in range(4):
                if U_str[i][r] != 0:
                    U[strassen_to_full_U[i]][col_offset + r] += U_str[i][r]
            for j in range(4):
                if V_str[j][r] != 0:
                    V[strassen_to_full_V[j]][col_offset + r] += V_str[j][r]
            for k in range(4):
                if W_str[k][r] != 0:
                    W[strassen_to_full_W[k]][col_offset + r] += W_str[k][r]
        
        # Schoolbook part: A_right (2x1) * B_bot (1x2)
        # A_right elements: a02, a12
        # a02: row 0, col 2 → U index 0*3+2 = 2
        # a12: row 1, col 2 → U index 1*3+2 = 5
        #
        # B_bot elements: b20, b21
        # b20: row 2, col block*2+0 → V index 2*6 + block*2+0
        # b21: row 2, col block*2+1 → V index 2*6 + block*2+1
        #
        # Output: same 2x2 C as above
        
        # Schoolbook U indices (for 2x1 A_right) to full U indices
        # Schoolbook U index 0 = a10 (row 0) → full U index 2
        # Schoolbook U index 1 = a20 (row 1) → full U index 5
        
        schoolbook_to_full_U = [2, 5]
        
        # Schoolbook V indices (for 1x2 B_bot) to full V indices
        # Schoolbook V index 0 = b10 (col 0) → full V index 2*6 + block*2+0
        # Schoolbook V index 1 = b11 (col 1) → full V index 2*6 + block*2+1
        
        schoolbook_to_full_V = [2*6 + block*2+0, 2*6 + block*2+1]
        
        # Schoolbook W indices (for 2x2 C) to full W indices - same as Strassen
        schoolbook_to_full_W = strassen_to_full_W
        
        # Add schoolbook terms
        for r in range(4):
            for i in range(2):
                if U_sb[i][r] != 0:
                    U[schoolbook_to_full_U[i]][col_offset + 7 + r] += U_sb[i][r]
            for j in range(2):
                if V_sb[j][r] != 0:
                    V[schoolbook_to_full_V[j]][col_offset + 7 + r] += V_sb[j][r]
            for k in range(4):
                if W_sb[k][r] != 0:
                    W[schoolbook_to_full_W[k]][col_offset + 7 + r] += W_sb[k][r]
    
    # Convert to integers where possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
