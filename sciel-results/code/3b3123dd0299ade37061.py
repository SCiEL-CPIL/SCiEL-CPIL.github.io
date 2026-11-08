"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Construct an explicit tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    # Rank target: 35 (better than schoolbook 40).
    #
    # Decomposition structure:
    #   - Split the middle dimension m=4 into m1=2 and m2=2.
    #   - Left part: A1 (2x2) * B1 (2x5)  -> use Strassen for 2x2x2 block (cols 0..3 of B1) 
    #     + schoolbook for 2x2x3 remainder (cols 4..6 of B1)? 
    #     Actually, let's use a known rank-11 decomposition for 2x2x3 and rank-7 for 2x2x2.
    #     Total for left part: 7 + 11 = 18.
    #   - Right part: A2 (2x2) * B2 (2x5)  -> same, 18.
    #   - Total rank: 18 + 18 = 36.
    #
    # However, we can do slightly better. Let's use rank 10 for 2x2x3 if available, or stick with 11.
    # Standard rank for 2x2x3 is 11. So total 36.
    #
    # Let's try to get rank 35 by finding a shared term or better split.
    # Alternatively, split p=5 into 2+3?
    #   Left: 2x4x2. Split m=4 into 2+2: two 2x2x2 blocks. Rank 7+7=14.
    #   Right: 2x4x3. Split m=4 into 2+2: two 2x2x3 blocks. Rank 11+11=22.
    #   Total: 14+22=36.
    #
    # Let's implement the rank 36 solution. It is robust and easier to verify.
    
    import numpy as np

    n, m, p = 2, 4, 5
    rank = 36
    
    # U: shape (n*m, rank) = (8, 36)
    # V: shape (m*p, rank) = (20, 36)
    # W: shape (p*n, rank) = (10, 36)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def strassen_222(U, V, W, offset):
        # Fill U, V, W for 2x2x2 Strassen decomposition starting at column 'offset'.
        # Indices in U: row = i*m + j, where i in 0..1, j in 0..1
        # Indices in V: row = j*p + k, where j in 0..1, k in 0..1
        # Indices in W: row = k*n + i, where k in 0..1, i in 0..1
        #
        # Strassen coefficients:
        # C11 = M5 + M1 - M4 - M6
        # C12 = M2 + M6
        # C21 = M3 + M4
        # C22 = M5 - M1 + M2 + M3
        #
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # A is 2x2, B is 2x2
        # A11 index in U: i=0, j=0 -> row 0
        # A12 index in U: i=0, j=1 -> row 1
        # A21 index in U: i=1, j=0 -> row 2
        # A22 index in U: i=1, j=1 -> row 3
        #
        # B11 index in V: j=0, k=0 -> row 0
        # B12 index in V: j=0, k=1 -> row 1
        # B21 index in V: j=1, k=0 -> row 2
        # B22 index in V: j=1, k=1 -> row 3
        #
        # C11 index in W: k=0, i=0 -> row 0
        # C12 index in W: k=0, i=1 -> row 1
        # C21 index in W: k=1, i=0 -> row 2
        # C22 index in W: k=1, i=1 -> row 3
        
        # M1
        U[0, offset] += 1; U[3, offset] += 1
        V[0, offset] += 1; V[1, offset] += 1
        W[0, offset] += 1; W[1, offset] -= 1; W[2, offset] -= 1; W[3, offset] -= 1
        
        # M2
        U[2, offset] += 1; U[3, offset] += 1
        V[0, offset] += 1
        W[1, offset] += 1; W[3, offset] += 1
        
        # M3
        U[0, offset] += 1
        V[2, offset] += 1; V[3, offset] -= 1
        W[2, offset] += 1; W[3, offset] += 1
        
        # M4
        U[3, offset] += 1
        V[1, offset] -= 1; V[0, offset] -= 1
        W[0, offset] += 1; W[2, offset] += 1
        
        # M5
        U[0, offset] += 1; U[1, offset] += 1
        V[3, offset] += 1
        W[0, offset] += 1; W[3, offset] -= 1
        
        # M6
        U[2, offset] -= 1; U[0, offset] -= 1
        V[0, offset] += 1; V[1, offset] += 1
        W[0, offset] += 1; W[1, offset] += 1
        
        # M7
        U[1, offset] -= 1; U[3, offset] -= 1
        V[2, offset] += 1; V[3, offset] += 1
        W[0, offset] -= 1; W[3, offset] += 1

    def schoolbook_223(U, V, W, offset):
        # Fill U, V, W for 2x2x3 schoolbook decomposition starting at column 'offset'.
        # 2x2x3 requires 2*2*3 = 12 terms? No, rank is 12 for schoolbook, but optimal is 11.
        # Let's just use schoolbook rank 12 to be safe and simple, or find rank 11.
        # Actually, for 2x2x3, rank is 11. Let's implement rank 11.
        # This is more complex. Let's just use schoolbook rank 12 for the 2x2x3 part.
        # Then total rank for 2x4x5 would be 2*(7+12) = 38.
        # Or 2*(11+11) = 44? No.
        # Let's stick to schoolbook for simplicity if we can't find rank 11 easily.
        # But 38 is worse than 40? No, 38 < 40.
        # Let's implement schoolbook for 2x2x3.
        
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    col = offset + (i*2 + j)*3 + k
                    U[i*2 + j, col] = 1
                    V[j*3 + k, col] = 1
                    W[k*2 + i, col] = 1

    # Left part: A1 (rows 0,1 of A) * B1 (cols 0,1 of B, which are rows 0,1 of B? No, B is 4x5)
    # A is 2x4, B is 4x5.
    # A1 is A[:, 0:2], A2 is A[:, 2:4].
    # B1 is B[0:2, :], B2 is B[2:4, :].
    
    # For A1 * B1 (2x2 * 2x5):
    # U rows: 0,1 (for A11, A12 where col index in A1 is 0,1)
    # V rows: 0,1,2,3,4 (for B1 rows 0,1 which map to V rows 0..4)
    # Wait, V index is j*p+k. For left part, j=0,1. So V rows 0..4.
    # W rows: k*n+i. For left part, k=0..4, i=0,1. So W rows 0..9.
    
    # Let's re-map indices carefully.
    # Global U: row = i*m + j. i in 0..1, j in 0..3.
    #   A1 corresponds to j=0,1. So U rows 0,1,2,3.
    #   A2 corresponds to j=2,3. So U rows 4,5,6,7.
    # Global V: row = j*p + k. j in 0..3, k in 0..4.
    #   B1 corresponds to j=0,1. So V rows 0..4 and 5..9.
    #   B2 corresponds to j=2,3. So V rows 10..14 and 15..19.
    # Global W: row = k*n + i. k in 0..4, i in 0..1.
    #   W rows are always 0..9.
    
    # So for left part (A1*B1), we use U rows 0..3, V rows 0..9, W rows 0..9.
    # For right part (A2*B2), we use U rows 4..7, V rows 10..19, W rows 0..9.
    
    # Left part:
    # Split B1 (2x5) into B1a (2x2) and B1b (2x3).
    # B1a: k=0,1. V rows 0,1 (for j=0) and 2,3 (for j=1)? 
    # Wait, V row = j*p+k.
    # For j=0: V rows 0,1,2,3,4 correspond to k=0..4.
    # For j=1: V rows 5,6,7,8,9 correspond to k=0..4.
    # So B1a (cols 0,1 of B1) uses V rows 0,1 (j=0,k=0,1) and 5,6 (j=1,k=0,1).
    # This is not contiguous in the Strassen sense because Strassen assumes B is 2x2 contiguous.
    # We can handle this by permuting or just setting values explicitly.
    
    # Let's write a generic Strassen setter that takes explicit row/col indices for U, V, W.
    
    def strassen_set(U, V, W, row_map_U, row_map_V, row_map_W, col_offset):
        # row_map_U: list of 4 ints, mapping A11,A12,A21,A22 to U rows.
        # row_map_V: list of 4 ints, mapping B11,B12,B21,B22 to V rows.
        # row_map_W: list of 4 ints, mapping C11,C12,C21,C22 to W rows.
        
        # M1
        U[row_map_U[0], col_offset] += 1
        U[row_map_U[3], col_offset] += 1
        V[row_map_V[0], col_offset] += 1
        V[row_map_V[1], col_offset] += 1
        W[row_map_W[0], col_offset] += 1
        W[row_map_W[1], col_offset] -= 1
        W[row_map_W[2], col_offset] -= 1
        W[row_map_W[3], col_offset] -= 1
        
        # M2
        U[row_map_U[2], col_offset] += 1
        U[row_map_U[3], col_offset] += 1
        V[row_map_V[0], col_offset] += 1
        W[row_map_W[1], col_offset] += 1
        W[row_map_W[3], col_offset] += 1
        
        # M3
        U[row_map_U[0], col_offset] += 1
        V[row_map_V[2], col_offset] += 1
        V[row_map_V[3], col_offset] -= 1
        W[row_map_W[2], col_offset] += 1
        W[row_map_W[3], col_offset] += 1
        
        # M4
        U[row_map_U[3], col_offset] += 1
        V[row_map_V[1], col_offset] -= 1
        V[row_map_V[0], col_offset] -= 1
        W[row_map_W[0], col_offset] += 1
        W[row_map_W[2], col_offset] += 1
        
        # M5
        U[row_map_U[0], col_offset] += 1
        U[row_map_U[1], col_offset] += 1
        V[row_map_V[3], col_offset] += 1
        W[row_map_W[0], col_offset] += 1
        W[row_map_W[3], col_offset] -= 1
        
        # M6
        U[row_map_U[2], col_offset] -= 1
        U[row_map_U[0], col_offset] -= 1
        V[row_map_V[0], col_offset] += 1
        V[row_map_V[1], col_offset] += 1
        W[row_map_W[0], col_offset] += 1
        W[row_map_W[1], col_offset] += 1
        
        # M7
        U[row_map_U[1], col_offset] -= 1
        U[row_map_U[3], col_offset] -= 1
        V[row_map_V[2], col_offset] += 1
        V[row_map_V[3], col_offset] += 1
        W[row_map_W[0], col_offset] -= 1
        W[row_map_W[3], col_offset] += 1

    # Left part A1 (rows 0,1 of A) -> U rows:
    # A11 (i=0,j=0) -> U[0]
    # A12 (i=0,j=1) -> U[1]
    # A21 (i=1,j=0) -> U[2]
    # A22 (i=1,j=1) -> U[3]
    row_map_U_left_1 = [0, 1, 2, 3]
    
    # B1a (rows 0,1 of B1, cols 0,1 of B) -> V rows:
    # B11 (j=0,k=0) -> V[0]
    # B12 (j=0,k=1) -> V[1]
    # B21 (j=1,k=0) -> V[5]
    # B22 (j=1,k=1) -> V[6]
    row_map_V_left_1 = [0, 1, 5, 6]
    
    # C (2x2) from B1a -> W rows:
    # C11 (i=0,k=0) -> W[0]
    # C12 (i=1,k=0) -> W[1]
    # C21 (i=0,k=1) -> W[2]
    # C22 (i=1,k=1) -> W[3]
    row_map_W_left_1 = [0, 1, 2, 3]
    
    strassen_set(U, V, W, row_map_U_left_1, row_map_V_left_1, row_map_W_left_1, 0)
    
    # B1b (rows 0,1 of B1, cols 2,3,4 of B) -> Schoolbook 2x2x3
    # U rows same: [0, 1, 2, 3]
    # V rows:
    # j=0: k=2,3,4 -> V[2], V[3], V[4]
    # j=1: k=2,3,4 -> V[7], V[8], V[9]
    # W rows:
    # k=2: i=0,1 -> W[4], W[5]
    # k=3: i=0,1 -> W[6], W[7]
    # k=4: i=0,1 -> W[8], W[9]
    
    # Schoolbook 2x2x3 uses 12 terms.
    # Total rank so far: 7 (Strassen) + 12 (Schoolbook) = 19.
    # This is getting large. 19*2 = 38.
    
    # Let's just use rank 11 for 2x2x3.
    # I will implement a known rank-11 decomposition for 2x2x3.
    # This is complex to write from scratch correctly under time pressure.
    # Given the constraints, I will return the schoolbook rank 40 solution which is safe,
    # or try a simpler block structure that yields rank 36 with schoolbook 2x2x1s.
    
    # Let's stick to the plan of rank 36 using 2x2x2 Strassen + 2x2x1 Schoolbook.
    # Reset arrays.
    U.fill(0)
    V.fill(0)
    W.fill(0)
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Left part (18 ranks)
    # A1 (2x2) * B1 (2x5)
    # Split B1 into B1a (2x2) and B1b (2x3).
    # Use Strassen for B1a (7 ranks).
    # Use Schoolbook for B1b (2x2x3 = 12 ranks). Total 19.
    # This exceeds 18.
    
    # Alternative: Split B1 into B1a (2x2), B1b (2x2), B1c (2x1).
    # 7 + 7 + 4 = 18.
    # Yes! Split p=5 into 2+2+1.
    
    # Left Part 1: A1 * B1a (cols 0,1) -> Strassen 7
    row_map_U_left_1 = [0, 1, 2, 3]
    row_map_V_left_1 = [0, 1, 5, 6] # B11, B12, B21, B22
    row_map_W_left_1 = [0, 1, 2, 3] # C11, C12, C21, C22
    strassen_set(U, V, W, row_map_U_left_1, row_map_V_left_1, row_map_W_left_1, 0)
    
    # Left Part 2: A1 * B1b (cols 2,3) -> Strassen 7
    # B1b: j=0, k=2,3 -> V[2], V[3]; j=1, k=2,3 -> V[7], V[8]
    row_map_V_left_2 = [2, 3, 7, 8]
    # C from B1b: k=2,3 -> W[4..7]
    row_map_W_left_2 = [4, 5, 6, 7]
    strassen_set(U, V, W, row_map_U_left_1, row_map_V_left_2, row_map_W_left_2, 7)
    
    # Left Part 3: A1 * B1c (col 4) -> Schoolbook 2x2x1 (4 ranks)
    # B1c: j=0, k=4 -> V[4]; j=1, k=4 -> V[9]
    # C from B1c: k=4 -> W[8], W[9]
    # Schoolbook 2x2x1:
    # i=0, j=0: U[0], V[4], W[8]
    # i=0, j=1: U[1], V[9], W[8]
    # i=1, j=0: U[2], V[4], W[9]
    # i=1, j=1: U[3], V[9], W[9]
    U[0, 14] = 1; V[4, 14] = 1; W[8, 14] = 1
    U[1, 15] = 1; V[9, 15] = 1; W[8, 15] = 1
    U[2, 16] = 1; V[4, 16] = 1; W[9, 16] = 1
    U[3, 17] = 1; V[9, 17] = 1; W[9, 17] = 1
    
    # Right Part (18 ranks)
    # A2 (rows 2,3 of A -> U rows 4..7) * B2 (rows 2,3 of B -> V rows 10..19)
    # Split B2 into B2a (cols 0,1), B2b (cols 2,3), B2c (col 4).
    
    # Right Part 1: A2 * B2a (cols 0,1) -> Strassen 7
    row_map_U_right_1 = [4, 5, 6, 7]
    # B2a: j=2, k=0,1 -> V[10], V[11]; j=3, k=0,1 -> V[15], V[16]
    row_map_V_right_1 = [10, 11, 15, 16]
    # C from B2a: k=0,1 -> W[0..3]
    row_map_W_right_1 = [0, 1, 2, 3]
    strassen_set(U, V, W, row_map_U_right_1, row_map_V_right_1, row_map_W_right_1, 18)
    
    # Right Part 2: A2 * B2b (cols 2,3) -> Strassen 7
    # B2b: j=2, k=2,3 -> V[12], V[13]; j=3, k=2,3 -> V[17], V[18]
    row_map_V_right_2 = [12, 13, 17, 18]
    # C from B2b: k=2,3 -> W[4..7]
    row_map_W_right_2 = [4, 5, 6, 7]
    strassen_set(U, V, W, row_map_U_right_1, row_map_V_right_2, row_map_W_right_2, 25)
    
    # Right Part 3: A2 * B2c (col 4) -> Schoolbook 2x2x1 (4 ranks)
    # B2c: j=2, k=4 -> V[14]; j=3, k=4 -> V[19]
    # C from B2c: k=4 -> W[8], W[9]
    U[4, 32] = 1; V[14, 32] = 1; W[8, 32] = 1
    U[5, 33] = 1; V[19, 33] = 1; W[8, 33] = 1
    U[6, 34] = 1; V[14, 34] = 1; W[9, 34] = 1
    U[7, 35] = 1; V[19, 35] = 1; W[9, 35] = 1
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
