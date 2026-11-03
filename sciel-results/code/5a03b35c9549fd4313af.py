# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs the tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses a block decomposition to achieve rank 46.
    1. Split intermediate dimension 5 into 2 and 3.
    2. A = [A1, A2] where A1 is 2x2, A2 is 2x3.
    3. B = [B1; B2] where B1 is 2x5, B2 is 3x5.
    4. C = A1*B1 + A2*B2.
    
    A1*B1 (2x2x5) decomposed into:
      - A1 * B1a (2x2x2) using Strassen (rank 7)
      - A1 * B1b (2x2x2) using Strassen (rank 7)
      - A1 * B1c (2x2x1) using Schoolbook (rank 4)
    Total rank for A1*B1: 18.
    
    A2*B2 (2x3x5) decomposed into:
      - A2a * B2a (2x1x5) using Schoolbook (rank 10)
      - A2b * B2b (2x2x5) using same as A1*B1 (rank 18)
    Total rank for A2*B2: 28.
    
    Total rank: 18 + 28 = 46.
    """
    n, m, p = 2, 5, 5
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)] # 10 x 46
    V = [[0] * rank for _ in range(m * p)] # 25 x 46
    W = [[0] * rank for _ in range(p * n)] # 10 x 46
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_2x2x2(u_offset, v_offset, w_offset, start_col):
        # U indices: u_offset + i*2 + j (i in 0..1, j in 0..1) -> 4 indices
        # V indices: v_offset + j*2 + k (j in 0..1, k in 0..1) -> 4 indices
        # W indices: w_offset + k*2 + i (k in 0..1, i in 0..1) -> 4 indices
        
        # Strassen mults:
        # m1 = (A11 + A22) * (B11 + B22)
        # m2 = (A21 + A22) * B11
        # m3 = A11 * (B12 - B22)
        # m4 = A22 * (B21 - B11)
        # m5 = (A11 + A12) * B22
        # m6 = (A21 - A11) * (B11 + B12)
        # m7 = (A12 - A22) * (B21 + B22)
        
        # We need to map these to U, V, W entries.
        # U[A_idx, col] = coeff
        # V[B_idx, col] = coeff
        # W[C_idx, col] = coeff
        
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # Let's add each multiplication
        # 1. (A11 + A22) * (B11 + B22)
        # U: A11(1), A22(1)
        # V: B11(1), B22(1)
        # W: C11(1), C22(1)
        U[u_offset + 0][start_col] = 1
        U[u_offset + 3][start_col] = 1
        V[v_offset + 0][start_col] = 1
        V[v_offset + 3][start_col] = 1
        W[w_offset + 0][start_col] = 1
        W[w_offset + 3][start_col] = 1
        start_col += 1
        
        # 2. (A21 + A22) * B11
        # U: A21(1), A22(1)
        # V: B11(1)
        # W: C21(1), C22(-1)
        U[u_offset + 2][start_col] = 1
        U[u_offset + 3][start_col] = 1
        V[v_offset + 0][start_col] = 1
        W[w_offset + 2][start_col] = 1
        W[w_offset + 3][start_col] = -1
        start_col += 1
        
        # 3. A11 * (B12 - B22)
        # U: A11(1)
        # V: B12(1), B22(-1)
        # W: C12(1), C22(1)
        U[u_offset + 0][start_col] = 1
        V[v_offset + 1][start_col] = 1
        V[v_offset + 3][start_col] = -1
        W[w_offset + 1][start_col] = 1
        W[w_offset + 3][start_col] = 1
        start_col += 1
        
        # 4. A22 * (B21 - B11)
        # U: A22(1)
        # V: B21(1), B11(-1)
        # W: C21(1), C22(-1)
        U[u_offset + 3][start_col] = 1
        V[v_offset + 2][start_col] = 1
        V[v_offset + 0][start_col] = -1
        W[w_offset + 2][start_col] = 1
        W[w_offset + 3][start_col] = -1
        start_col += 1
        
        # 5. (A11 + A12) * B22
        # U: A11(1), A12(1)
        # V: B22(1)
        # W: C11(-1), C12(1)
        U[u_offset + 0][start_col] = 1
        U[u_offset + 1][start_col] = 1
        V[v_offset + 3][start_col] = 1
        W[w_offset + 0][start_col] = -1
        W[w_offset + 1][start_col] = 1
        start_col += 1
        
        # 6. (A21 - A11) * (B11 + B12)
        # U: A21(1), A11(-1)
        # V: B11(1), B12(1)
        # W: C22(1)
        U[u_offset + 2][start_col] = 1
        U[u_offset + 0][start_col] = -1
        V[v_offset + 0][start_col] = 1
        V[v_offset + 1][start_col] = 1
        W[w_offset + 3][start_col] = 1
        start_col += 1
        
        # 7. (A12 - A22) * (B21 + B22)
        # U: A12(1), A22(-1)
        # V: B21(1), B22(1)
        # W: C11(1), C22(1)
        U[u_offset + 1][start_col] = 1
        U[u_offset + 3][start_col] = -1
        V[v_offset + 2][start_col] = 1
        V[v_offset + 3][start_col] = 1
        W[w_offset + 0][start_col] = 1
        W[w_offset + 3][start_col] = 1
        start_col += 1
        
        return start_col

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(u_offset, v_offset, w_offset, start_col):
        # A is 2x2, B is 2x1, C is 2x1
        # C1 = A11*B11 + A12*B21
        # C2 = A21*B11 + A22*B21
        
        # 1. A11 * B11
        U[u_offset + 0][start_col] = 1
        V[v_offset + 0][start_col] = 1
        W[w_offset + 0][start_col] = 1
        start_col += 1
        
        # 2. A12 * B21
        U[u_offset + 1][start_col] = 1
        V[v_offset + 1][start_col] = 1
        W[w_offset + 0][start_col] = 1
        start_col += 1
        
        # 3. A21 * B11
        U[u_offset + 2][start_col] = 1
        V[v_offset + 0][start_col] = 1
        W[w_offset + 1][start_col] = 1
        start_col += 1
        
        # 4. A22 * B21
        U[u_offset + 3][start_col] = 1
        V[v_offset + 1][start_col] = 1
        W[w_offset + 1][start_col] = 1
        start_col += 1
        
        return start_col

    # Helper to add Schoolbook 2x1x5 block
    def add_schoolbook_2x1x5(u_offset, v_offset, w_offset, start_col):
        # A is 2x1 (vector u), B is 1x5 (vector v), C is 2x5
        # C_i,j = u_i * v_j
        # Rank 10.
        for i in range(2):
            for j in range(5):
                U[u_offset + i][start_col] = 1
                V[v_offset + j][start_col] = 1
                W[w_offset + j * 2 + i][start_col] = 1
                start_col += 1
        return start_col

    # Indices mapping:
    # A is 2x5.
    # A1 is A[:, 0:2] -> indices 0,1,2,3 in U (i*2+j)
    # A2 is A[:, 2:5] -> indices 4,5,6,7,8,9 in U
    #   A2a is A[:, 2:3] -> indices 4,5
    #   A2b is A[:, 3:5] -> indices 6,7,8,9
    
    # B is 5x5.
    # B1 is B[0:2, :] -> indices 0..9 in V (j*5+k)
    #   B1a is B[0:2, 0:2] -> indices 0,1,5,6
    #   B1b is B[0:2, 2:4] -> indices 2,3,7,8
    #   B1c is B[0:2, 4:5] -> indices 4,9
    # B2 is B[2:5, :] -> indices 10..24 in V
    #   B2a is B[2:3, :] -> indices 10,11,12,13,14
    #   B2b is B[3:5, :] -> indices 15..24
    #     B2b_1 is B[3:5, 0:2] -> indices 15,16,20,21
    #     B2b_2 is B[3:5, 2:4] -> indices 17,18,22,23
    #     B2b_3 is B[3:5, 4:5] -> indices 19,24

    # C is 2x5.
    # W indices: k*2 + i (k is row of C? No, k is column of B / row of C?
    # Task says W[k*n+i, r].
    # C is 2x5. Rows i=0,1. Cols k=0..4.
    # So index is k*2 + i.
    # C1 corresponds to A1*B1.
    # C2 corresponds to A2*B2.
    # They add up to C.
    # So W entries for C1 and C2 go to the same W indices.

    # --- Part 1: A1 * B1 (2x2x5) ---
    # A1 indices in U: 0,1,2,3
    # B1 indices in V:
    #   B1a (cols 0,1): 0,1,5,6
    #   B1b (cols 2,3): 2,3,7,8
    #   B1c (col 4): 4,9
    
    # 1. A1 * B1a (2x2x2)
    # U offset 0
    # V offset 0 (B11=0, B12=1, B21=5, B22=6)
    # W offset 0 (C11=0, C12=1, C21=2, C22=3) -- Wait.
    # C1 is 2x2.
    # C1[0,0] -> W[0*2+0]=0
    # C1[0,1] -> W[1*2+0]=2
    # C1[1,0] -> W[0*2+1]=1
    # C1[1,1] -> W[1*2+1]=3
    
    # My Strassen helper assumes:
    # U: 0,1,2,3 (A11, A12, A21, A22)
    # V: 0,1,2,3 (B11, B12, B21, B22)
    # W: 0,1,2,3 (C11, C12, C21, C22)
    
    # For B1a:
    # B11 (B[0,0]) -> V[0]
    # B12 (B[0,1]) -> V[1]
    # B21 (B[1,0]) -> V[5]
    # B22 (B[1,1]) -> V[6]
    # So V indices are 0, 1, 5, 6.
    # But my helper uses consecutive offsets.
    # I need to adapt the helper or map indices.
    # Let's modify the helper to take explicit index lists or offsets that are not necessarily contiguous in the global array?
    # Easier: Just write the specific multiplications for each block.
    
    # Let's define a generic add_mult(u_idx, v_idx, w_idx, sign_u, sign_v, sign_w, col)
    # But Strassen has sums.
    
    # Let's just inline the Strassen for B1a.
    # U_A1 = [0, 1, 2, 3]
    # V_B1a = [0, 1, 5, 6]
    # W_C1a = [0, 2, 1, 3] (C11, C12, C21, C22)
    
    # m1 = (A11+A22)(B11+B22)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # m2 = (A21+A22)B11
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # m3 = A11(B12-B22)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # m4 = A22(B21-B11)
    U[3][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # m5 = (A11+A12)B22
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # m6 = (A21-A11)(B11+B12)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # m7 = (A12-A22)(B21+B22)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # 2. A1 * B1b (2x2x2)
    # U_A1 = [0, 1, 2, 3]
    # V_B1b = [2, 3, 7, 8]
    # W_C1b = [4, 6, 5, 7] (C[0,2], C[0,3], C[1,2], C[1,3])
    # W indices:
    # C[0,2] -> k=2, i=0 -> 2*2+0=4
    # C[0,3] -> k=3, i=0 -> 3*2+0=6
    # C[1,2] -> k=2, i=1 -> 2*2+1=5
    # C[1,3] -> k=3, i=1 -> 3*2+1=7
    
    # m1
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # m2
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # m3
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # m4
    U[3][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # m5
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    # m6
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # m7
    U[1][col] = 1; U[3][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # 3. A1 * B1c (2x2x1)
    # U_A1 = [0, 1, 2, 3]
    # V_B1c = [4, 9]
    # W_C1c = [8, 9] (C[0,4], C[1,4])
    # C[0,4] -> k=4, i=0 -> 4*2+0=8
    # C[1,4] -> k=4, i=1 -> 4*2+1=9
    
    # 1. A11*B11
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    # 2. A12*B21
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # 3. A21*B11
    U[2][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    # 4. A22*B21
    U[3][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # --- Part 2: A2 * B2 (2x3x5) ---
    # A2 is 2x3.
    # A2a is A[:, 2:3] -> U[4], U[5]
    # A2b is A[:, 3:5] -> U[6], U[7], U[8], U[9]
    
    # B2 is 3x5.
    # B2a is B[2:3, :] -> V[10..14]
    # B2b is B[3:5, :] -> V[15..24]
    #   B2b_1 (cols 0,1) -> V[15,16,20,21]
    #   B2b_2 (cols 2,3) -> V[17,18,22,23]
    #   B2b_3 (col 4) -> V[19,24]
    
    # 1. A2a * B2a (2x1x5)
    # U_A2a = [4, 5]
    # V_B2a = [10, 11, 12, 13, 14]
    # W_C2a = [4, 6, 8, 10, 12] (C[0,0], C[0,2], C[0,4], C[1,0], C[1,2], C[1,4]?)
    # Wait. C is 2x5.
    # C[0,0] -> W[0]
    # C[0,1] -> W[2]
    # C[0,2] -> W[4]
    # C[0,3] -> W[6]
    # C[0,4] -> W[8]
    # C[1,0] -> W[1]
    # C[1,1] -> W[3]
    # C[1,2] -> W[5]
    # C[1,3] -> W[7]
    # C[1,4] -> W[9]
    
    # A2a * B2a contributes to ALL columns of C?
    # A2a is col 2 of A. B2a is row 2 of B.
    # Product is A[0,2]*B[2,k] and A[1,2]*B[2,k].
    # This contributes to C[0,k] and C[1,k] for all k.
    # So W indices are 0,2,4,6,8 for i=0 and 1,3,5,7,9 for i=1.
    
    for j in range(5):
        # u_0 * v_j
        U[4][col] = 1
        V[10 + j][col] = 1
        W[j * 2 + 0][col] = 1
        col += 1
        # u_1 * v_j
        U[5][col] = 1
        V[10 + j][col] = 1
        W[j * 2 + 1][col] = 1
        col += 1
        
    # 2. A2b * B2b (2x2x5)
    # Same as A1 * B1 but with different indices.
    # U_A2b = [6, 7, 8, 9]
    # V_B2b_1 = [15, 16, 20, 21]
    # W_C2b_1 = [0, 2, 1, 3]
    
    # m1
    U[6][col] = 1; U[9][col] = 1
    V[15][col] = 1; V[21][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # m2
    U[8][col] = 1; U[9][col] = 1
    V[15][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # m3
    U[6][col] = 1
    V[16][col] = 1; V[21][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # m4
    U[9][col] = 1
    V[20][col] = 1; V[15][col] = -1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # m5
    U[6][col] = 1; U[7][col] = 1
    V[21][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # m6
    U[8][col] = 1; U[6][col] = -1
    V[15][col] = 1; V[16][col] = 1
    W[3][col] = 1
    col += 1
    
    # m7
    U[7][col] = 1; U[9][col] = -1
    V[20][col] = 1; V[21][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # 3. A2b * B2b_2 (2x2x2)
    # V_B2b_2 = [17, 18, 22, 23]
    # W_C2b_2 = [4, 6, 5, 7]
    
    # m1
    U[6][col] = 1; U[9][col] = 1
    V[17][col] = 1; V[23][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # m2
    U[8][col] = 1; U[9][col] = 1
    V[17][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # m3
    U[6][col] = 1
    V[18][col] = 1; V[23][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # m4
    U[9][col] = 1
    V[22][col] = 1; V[17][col] = -1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # m5
    U[6][col] = 1; U[7][col] = 1
    V[23][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    # m6
    U[8][col] = 1; U[6][col] = -1
    V[17][col] = 1; V[18][col] = 1
    W[7][col] = 1
    col += 1
    
    # m7
    U[7][col] = 1; U[9][col] = -1
    V[22][col] = 1; V[23][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # 4. A2b * B2b_3 (2x2x1)
    # V_B2b_3 = [19, 24]
    # W_C2b_3 = [8, 9]
    
    # 1. A11*B11
    U[6][col] = 1
    V[19][col] = 1
    W[8][col] = 1
    col += 1
    
    # 2. A12*B21
    U[7][col] = 1
    V[24][col] = 1
    W[8][col] = 1
    col += 1
    
    # 3. A21*B11
    U[8][col] = 1
    V[19][col] = 1
    W[9][col] = 1
    col += 1
    
    # 4. A22*B21
    U[9][col] = 1
    V[24][col] = 1
    W[9][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
