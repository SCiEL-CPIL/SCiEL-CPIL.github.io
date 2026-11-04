"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # We'll use a block decomposition approach.
    # Split the 9 columns of B into 3 blocks of 3 columns each.
    # For each block (2x3 * 3x3), we'll use a decomposition of rank 15.
    # Total rank = 3 * 15 = 45.
    
    # For a 2x3 * 3x3 multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # 
    # We can split A into A_top (2x2) and A_bottom (2x1), and B into B_top (3x2) and B_bottom (3x1)
    # Wait, let me think more carefully.
    #
    # Actually, let's use a different approach. For 2x3 * 3x3:
    # We can view this as computing 2 outputs, each being a dot product of length 3.
    # But we can use Strassen-like ideas or just a clever decomposition.
    #
    # Let me try a direct construction. For 2x3 * 3x3, the schoolbook rank is 18.
    # Can we do better?
    #
    # Actually, let me try a different block structure. 
    # Split the middle dimension (3) into 2+1.
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x3); B_bottom (1x3)]
    # 
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top is 2x2 * 2x3, which has schoolbook rank 12.
    # A_right * B_bottom is 2x1 * 1x3, which has rank 2 (outer product).
    # Total: 12 + 2 = 14.
    #
    # But wait, can we do better than 14 for 2x2 * 2x3?
    # 2x2 * 2x2 has rank 7 (Strassen).
    # 2x2 * 2x3 can be split into 2x2 * 2x2 and 2x2 * 2x1.
    # 2x2 * 2x2 has rank 7.
    # 2x2 * 2x1 has rank 4 (schoolbook) or can it be done in less?
    # Actually, 2x2 * 2x1: C[i] = sum_j A[i,j] * B[j]. This is just two dot products.
    # Rank is 4 (each output is a sum of 2 products, but they share inputs).
    # Actually, the rank of 2x2*2x1 is 4 because we need 4 multiplications.
    # Hmm, but maybe we can do better.
    #
    # Let me just use rank 14 for 2x3*3x3 and see if that works.
    # Total rank = 3 * 14 = 42.
    
    # Actually, let me be more careful and construct the decomposition explicitly.
    
    # For 2x2 * 2x2, Strassen's algorithm gives rank 7.
    # The 7 products are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Let me encode this. For 2x2*2x2, we need U, V, W of shapes (4,7), (4,7), (4,7).
    # U is indexed by (i,j) for A, V by (j,k) for B, W by (k,i) for C.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11 = U[0], A22 = U[3], B11 = V[0], B22 = V[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 += M1, C22 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21 + A22)B11
        # A21 = U[2], A22 = U[3], B11 = V[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += M2
        W[2][1] = 1
        
        # M3 = A11(B21 - B22)
        # A11 = U[0], B21 = V[2], B22 = V[3]
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        # C12 += M3, C22 += M3
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A22(B11 - B12)
        # A22 = U[3], B11 = V[0], B12 = V[1]
        U[3][3] = 1
        V[0][3] = 1; V[1][3] = -1
        # C11 -= M4, C21 += M4
        W[0][3] = -1; W[2][3] = 1
        
        # M5 = (A11 + A12)B22
        # A11 = U[0], A12 = U[1], B22 = V[3]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11 += M5, C12 += M5
        W[0][4] = 1; W[1][4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        # A21 = U[2], A11 = U[0], B11 = V[0], B12 = V[1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 += M6
        W[3][5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # A12 = U[1], A22 = U[3], B21 = V[2], B22 = V[3]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 -= M7
        W[0][6] = -1
        
        return U, V, W
    
    # For 2x2 * 2x1 (rank 4, schoolbook):
    # C[i] = sum_j A[i,j] * B[j]
    # This is just 4 multiplications.
    def schoolbook_2x2x1():
        """Return U, V, W for 2x2 * 2x1 with rank 4."""
        U = [[0.0]*4 for _ in range(4)]
        V = [[0.0]*4 for _ in range(2)]
        W = [[0.0]*4 for _ in range(2)]
        
        for i in range(2):
            for j in range(2):
                idx = i*2 + j
                U[idx][idx] = 1
                V[j][idx] = 1
                W[i][idx] = 1
        
        return U, V, W
    
    # For 2x1 * 1x2 (rank 2):
    # C[i,k] = A[i] * B[k]
    def outer_2x1x2():
        """Return U, V, W for 2x1 * 1x2 with rank 2."""
        U = [[0.0]*2 for _ in range(2)]
        V = [[0.0]*2 for _ in range(2)]
        W = [[0.0]*2 for _ in range(4)]
        
        # A has 2 entries, B has 2 entries, C has 4 entries
        # C[i,k] = A[i] * B[k]
        # We need: sum_c U[i,c] * V[k,c] * W[k*2+i,c] = A[i]*B[k]
        # Let's use 2 terms:
        # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1, W[2,0]=1 (C[0,0] and C[1,0])
        # Wait, W is indexed by (k,i) with shape (p*n, rank) = (2*2, 2) = (4, 2)
        # W[k*n + i, c]
        
        U[0][0] = 1  # A[0]
        U[1][1] = 1  # A[1]
        V[0][0] = 1  # B[0]
        V[1][1] = 1  # B[1]
        # W[k*2 + i, c]
        # For term 0: A[0]*B[0] -> C[0,0]
        W[0*2 + 0][0] = 1  # C[0,0]
        # For term 1: A[1]*B[1] -> C[1,1]
        W[1*2 + 1][1] = 1  # C[1,1]
        # But we also need C[0,1] = A[0]*B[1] and C[1,0] = A[1]*B[0]
        # This doesn't work with just 2 terms. We need 4 terms for the full outer product.
        # Actually, the rank of 2x1*1x2 is 4, not 2.
        
        # Let me redo this. The rank of m x 1 * 1 x p is m*p.
        # So 2x1*1x2 has rank 4.
        
        U = [[0.0]*4 for _ in range(2)]
        V = [[0.0]*4 for _ in range(2)]
        W = [[0.0]*4 for _ in range(4)]
        
        for i in range(2):
            for k in range(2):
                idx = i*2 + k
                U[i][idx] = 1
                V[k][idx] = 1
                W[k*2 + i][idx] = 1
        
        return U, V, W
    
    # Now for 2x3 * 3x3:
    # Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_top (2x3) and B_bottom (1x3)
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top is 2x2 * 2x3.
    # Split B_top into B_top_left (2x2) and B_top_right (2x1).
    # A_left * B_top_left is 2x2 * 2x2 (rank 7).
    # A_left * B_top_right is 2x2 * 2x1 (rank 4).
    # Total for A_left * B_top: 7 + 4 = 11.
    #
    # A_right * B_bottom is 2x1 * 1x3 (rank 6).
    #
    # Total for 2x3 * 3x3: 11 + 6 = 17.
    #
    # Hmm, that's worse than my earlier estimate. Let me try a different split.
    #
    # Actually, let me just use schoolbook for the 2x3*3x3 blocks.
    # Schoolbook rank for 2x3*3x3 is 18.
    # Total rank = 3 * 18 = 54. That's the same as the original.
    #
    # Let me try yet another approach. What if I don't block at all?
    # The full problem is 2x3 * 3x9.
    # Schoolbook rank is 54.
    #
    # Let me try to use Strassen on a 2x2 subproblem.
    # Split the 3 middle dimension into 2+1.
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x9); B2 (1x9)]
    # C = A1 * B1 + A2 * B2
    #
    # A1 * B1 is 2x2 * 2x9.
    # Split B1 into 9 columns of 2x1.
    # Each 2x2 * 2x1 has rank 4.
    # Total for A1 * B1: 9 * 4 = 36.
    #
    # A2 * B2 is 2x1 * 1x9 (rank 18).
    #
    # Total: 36 + 18 = 54. Still the same.
    #
    # Let me try to use Strassen more aggressively.
    # For 2x2 * 2x9, I can group the 9 columns into 4+4+1.
    # 2x2 * 2x4 can be done by applying Strassen to 2x2*2x2 twice, giving 2*7 = 14.
    # 2x2 * 2x1 has rank 4.
    # Total for 2x2 * 2x9: 14 + 14 + 4 = 32.
    #
    # A2 * B2 is 2x1 * 1x9 (rank 18).
    #
    # Total: 32 + 18 = 50.
    #
    # That's better! 50 < 54.
    
    # Let me implement this.
    # 
    # For 2x2 * 2x2, use Strassen (rank 7).
    # For 2x2 * 2x1, use schoolbook (rank 4).
    # For 2x1 * 1x9, use schoolbook (rank 18).
    
    # Let me construct the full decomposition.
    
    # A is 2x3, B is 3x9.
    # A indices: (i,j) -> i in {0,1}, j in {0,1,2}. Flat index: i*3+j.
    # B indices: (j,k) -> j in {0,1,2}, k in {0,...,8}. Flat index: j*9+k.
    # C indices: (i,k) -> i in {0,1}, k in {0,...,8}. Flat index: k*2+i (W[k*n+i]).
    
    # Split A into A1 (2x2, j=0,1) and A2 (2x1, j=2).
    # Split B into B1 (2x9, j=0,1) and B2 (1x9, j=2).
    
    # C = A1 * B1 + A2 * B2
    
    # A1 * B1: 2x2 * 2x9
    # Split B1 into B1a (2x4, k=0..3), B1b (2x4, k=4..7), B1c (2x1, k=8)
    # A1 * B1a: 2x2 * 2x4, use two Strassen blocks (each 2x2*2x2, rank 7) -> rank 14
    # A1 * B1b: 2x2 * 2x4, use two Strassen blocks (each 2x2*2x2, rank 7) -> rank 14
    # A1 * B1c: 2x2 * 2x1, schoolbook -> rank 4
    # Total for A1 * B1: 14 + 14 + 4 = 32
    
    # A2 * B2: 2x1 * 1x9, schoolbook -> rank 18
    
    # Total rank: 32 + 18 = 50
    
    # Let me construct this.
    
    # First, let me create the Strassen 2x2*2x2 decomposition.
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Schoolbook 2x2*2x1
    U_sb, V_sb, W_sb = schoolbook_2x2x1()
    
    # Schoolbook 2x1*1x9
    # A2 has 2 entries, B2 has 9 entries, C has 18 entries.
    # C[i,k] = A2[i] * B2[k]
    # U: (2, 18), V: (9, 18), W: (18, 18)
    U_outer = [[0.0]*18 for _ in range(2)]
    V_outer = [[0.0]*18 for _ in range(9)]
    W_outer = [[0.0]*18 for _ in range(18)]
    for i in range(2):
        for k in range(9):
            idx = i*9 + k
            U_outer[i][idx] = 1
            V_outer[k][idx] = 1
            W_outer[k*2 + i][idx] = 1
    
    # Now let me assemble the full decomposition.
    # Total rank = 50.
    # 
    # The decomposition has 3 parts:
    # Part 1: A1 * B1a (2x2 * 2x4), rank 14
    # Part 2: A1 * B1b (2x2 * 2x4), rank 14
    # Part 3: A1 * B1c (2x2 * 2x1), rank 4
    # Part 4: A2 * B2 (2x1 * 1x9), rank 18
    
    # Let me create the full U, V, W matrices.
    rank = 50
    U_full = [[0.0]*rank for _ in range(6)]  # 2*3 = 6
    V_full = [[0.0]*rank for _ in range(27)]  # 3*9 = 27
    W_full = [[0.0]*rank for _ in range(18)]  # 9*2 = 18
    
    # Part 1: A1 * B1a
    # A1 is A[:, 0:2], B1a is B[0:2, 0:4]
    # A1 indices: (i,0) -> i*3+0, (i,1) -> i*3+1
    # B1a indices: (0,k) -> 0*9+k = k, (1,k) -> 1*9+k = 9+k for k=0..3
    # C indices: (i,k) -> k*2+i for i=0,1, k=0..3
    
    # Use two Strassen blocks for the 2x4.
    # Strassen block 1: columns 0,1
    # Strassen block 2: columns 2,3
    
    col_offset = 0
    for s in range(2):  # Two Strassen blocks
        for c in range(7):  # 7 columns in Strassen
            col = col_offset + c
            # U: A1 entries
            # U_s has shape (4,7), indexed by (i,j) for i,j in {0,1}
            # A1[i,j] is at flat index i*3 + j
            for ij in range(4):
                i_s, j_s = divmod(ij, 2)
                U_full[i_s*3 + j_s][col] = U_s[ij][c]
            
            # V: B1a entries
            # V_s has shape (4,7), indexed by (j,k) for j,k in {0,1}
            # B1a[j,k] is at flat index j*9 + k for j=0,1, k=0..3
            # For Strassen block s, k_offset = s*2
            for jk in range(4):
                j_s, k_s = divmod(jk, 2)
                k_actual = s*2 + k_s
                V_full[j_s*9 + k_actual][col] = V_s[jk][c]
            
            # W: C entries
            # W_s has shape (4,7), indexed by (k,i) for k,i in {0,1}
            # C[i,k] is at flat index k*2 + i for i=0,1, k=0..3
            # For Strassen block s, k_offset = s*2
            for ki in range(4):
                k_s, i_s = divmod(ki, 2)
                k_actual = s*2 + k_s
                W_full[k_actual*2 + i_s][col] = W_s[ki][c]
        
        col_offset += 7
    
    # Part 2: A1 * B1b
    # B1b is B[0:2, 4:8]
    # Similar to Part 1 but with k offset 4.
    
    for s in range(2):  # Two Strassen blocks
        for c in range(7):
            col = col_offset + c
            for ij in range(4):
                i_s, j_s = divmod(ij, 2)
                U_full[i_s*3 + j_s][col] = U_s[ij][c]
            
            for jk in range(4):
                j_s, k_s = divmod(jk, 2)
                k_actual = 4 + s*2 + k_s
                V_full[j_s*9 + k_actual][col] = V_s[jk][c]
            
            for ki in range(4):
                k_s, i_s = divmod(ki, 2)
                k_actual = 4 + s*2 + k_s
                W_full[k_actual*2 + i_s][col] = W_s[ki][c]
        
        col_offset += 7
    
    # Part 3: A1 * B1c
    # B1c is B[0:2, 8:9]
    # Use schoolbook 2x2*2x1 (rank 4)
    
    for c in range(4):
        col = col_offset + c
        for ij in range(4):
            i_s, j_s = divmod(ij, 2)
            U_full[i_s*3 + j_s][col] = U_sb[ij][c]
        
        for j in range(2):
            V_full[j*9 + 8][col] = V_sb[j][c]
        
        for i in range(2):
            W_full[8*2 + i][col] = W_sb[i][c]
    
    col_offset += 4
    
    # Part 4: A2 * B2
    # A2 is A[:, 2], B2 is B[2, :]
    # U_outer has shape (2, 18), V_outer has shape (9, 18), W_outer has shape (18, 18)
    
    for c in range(18):
        col = col_offset + c
        for i in range(2):
            U_full[i*3 + 2][col] = U_outer[i][c]
        
        for k in range(9):
            V_full[2*9 + k][col] = V_outer[k][c]
        
        for ki in range(18):
            W_full[ki][col] = W_outer[ki][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
