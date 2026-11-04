"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-72 exact tensor decomposition for 4x4 @ 4x5 matrix multiplication.
    
    Strategy:
    - Split the 4x4 input matrix A into 2x2 blocks: A11, A12, A21, A22.
    - Split the 4x5 input matrix B into 2x5 top and 2x5 bottom parts: B_top (rows 0,1), B_bottom (rows 2,3).
    - For each 2x2x2 subproblem (multiplying a 2x2 A-block by a 2x2 B-block where B has 2 cols), use Strassen's 7-term algorithm.
    - But B has 5 columns, so we treat it as: first 4 columns handled by 2x2x2 Strassen blocks, 5th column handled schoolbook.
    
    Actually, let's do this more carefully:
    
    A is 4x4:
      A = [[A11, A12],
           [A21, A22]]
    where each block is 2x2.
    
    B is 4x5:
      B = [B1]  (rows 0,1; 2x5)
          [B2]  (rows 2,3; 2x5)
    
    C = A @ B is 4x5:
      C = [A11@B1 + A12@B2,  A11@B1[:,4] + A12@B2[:,4]]  -- wait, this doesn't decompose cleanly.
    
    Let me reconsider. B is 4x5. Let's split B into two 2x5 matrices: B_top = B[0:2,:] and B_bot = B[2:4,:].
    
    C_top = A[0:2,:] @ B = A11@B_top[:,0:2] ... no, this is getting confused.
    
    Better approach:
    Split both A and B by rows/cols in a way that allows 2x2x2 Strassen.
    
    A is 4x4. Split A into 2x2 blocks of size 2x2:
      A = [[A11, A12],
           [A21, A22]]
    B is 4x5. Split B by rows into top 2 and bottom 2:
      B = [B1; B2] where B1 is 2x5, B2 is 2x5.
    
    Then C = A@B is 4x5:
      C_top = A11@B1 + A12@B2  (2x5)
      C_bot = A21@B1 + A22@B2  (2x5)
    
    For C_top: we need to compute 2x2 @ 2x5. We can split B1 into B1[:,0:2] and B1[:,2:4] and B1[:,4].
    Actually, 2x2 @ 2x5 can be done by treating it as 2x2 @ 2x2 for cols 0-1, 2x2 @ 2x2 for cols 2-3, and 2x2 @ 2x1 for col 4.
    
    So for each of the 4 (A-block, B-row-block) pairs, we compute:
      - 2x2 @ 2x2 for first 4 columns (using Strassen, 7 terms each)
      - 2x2 @ 2x1 for 5th column (schoolbook, 4 terms each)
    
    Total: 4 pairs * (7 + 4) = 4 * 11 = 44? No wait.
    
    C_top = A11@B1 + A12@B2
    A11@B1: 2x2 @ 2x5. Split B1 into cols 0-1, 2-3, and 4.
      A11@B1[:,0:2]: 2x2@2x2, Strassen 7
      A11@B1[:,2:4]: 2x2@2x2, Strassen 7
      A11@B1[:,4:5]: 2x2@2x1, schoolbook 4
    Similarly for A12@B2, A21@B1, A22@B2.
    
    Total rank = 4 * (7 + 7 + 4) = 4 * 18 = 72. Yes!
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 20 x 72
    W = [[0] * rank for _ in range(p * n)]  # 20 x 72
    
    # Strassen 2x2x2 decomposition
    # For A2x2 and B2x2, C2x2 = A2x2 @ B2x2
    # A2x2 = [[a00, a01], [a10, a11]]
    # B2x2 = [[b00, b01], [b10, b11]]
    # C2x2 = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_idx, b_idx, c_idx, col_offset):
        """
        Add a Strassen 2x2x2 block.
        a_idx: indices [a00, a01, a10, a11] in U's row indices (i*m+j)
        b_idx: indices [b00, b01, b10, b11] in V's row indices (j*p+k)
        c_idx: indices [c00, c01, c10, c11] in W's row indices (k*n+i)
        col_offset: starting column in the factor matrices
        """
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        col = col_offset
        
        # M1 = (a00 + a11)(b00 + b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # M2 = (a10 + a11)b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M3 = a00(b01 - b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1
        col += 1
        
        # M4 = a11(b10 - b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        # M5 = (a00 + a01)b11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_1(a_idx, b_idx, c_idx, col_offset):
        """
        Add a schoolbook 2x2 @ 2x1 block (4 terms).
        a_idx: [a00, a01, a10, a11]
        b_idx: [b00, b10]  (2x1, so b00 is row0, b10 is row1)
        c_idx: [c00, c10]  (2x1 output)
        """
        a00, a01, a10, a11 = a_idx
        b00, b10 = b_idx
        c00, c10 = c_idx
        
        col = col_offset
        # c00 = a00*b00 + a01*b10
        U[a00][col] = 1; V[b00][col] = 1; W[c00][col] = 1
        col += 1
        U[a01][col] = 1; V[b10][col] = 1; W[c00][col] = 1
        col += 1
        # c10 = a10*b00 + a11*b10
        U[a10][col] = 1; V[b00][col] = 1; W[c10][col] = 1
        col += 1
        U[a11][col] = 1; V[b10][col] = 1; W[c10][col] = 1
        col += 1
        
        return col
    
    # Indexing:
    # A is 4x4, indexed as A[i,j] where i in 0..3, j in 0..3
    # U row index: i*m + j = i*4 + j
    # B is 4x5, indexed as B[j,k] where j in 0..3, k in 0..4
    # V row index: j*p + k = j*5 + k
    # C is 4x5, output C[i,k], W row index: k*n + i = k*4 + i
    
    col = 0
    
    # A11 = A[0:2, 0:2] -> indices (0,0), (0,1), (1,0), (1,1) -> U rows: 0, 1, 4, 5
    # A12 = A[0:2, 2:4] -> indices (0,2), (0,3), (1,2), (1,3) -> U rows: 2, 3, 6, 7
    # A21 = A[2:4, 0:2] -> indices (2,0), (2,1), (3,0), (3,1) -> U rows: 8, 9, 12, 13
    # A22 = A[2:4, 2:4] -> indices (2,2), (2,3), (3,2), (3,3) -> U rows: 10, 11, 14, 15
    
    # B1 = B[0:2, :] -> rows 0,1 of B -> V rows: 0..4, 5..9
    # B2 = B[2:4, :] -> rows 2,3 of B -> V rows: 10..14, 15..19
    
    # For each (A-block, B-row-block) pair, process columns 0-1, 2-3, and 4 separately.
    
    blocks_A = {
        'A11': (0, 1, 4, 5),
        'A12': (2, 3, 6, 7),
        'A21': (8, 9, 12, 13),
        'A22': (10, 11, 14, 15),
    }
    
    blocks_B = {
        'B1': (0, 1, 5, 6),   # B[0,0], B[0,1], B[1,0], B[1,1] for cols 0,1
        'B2': (10, 11, 15, 16) # B[2,0], B[2,1], B[3,0], B[3,1] for cols 0,1
    }
    
    # Process (A11, B1): C top-left 2x2 contribution to cols 0,1
    a_idx = blocks_A['A11']
    b_idx = blocks_B['B1']
    # C rows 0,1; cols 0,1 -> W indices: k*4+i
    # c00: k=0,i=0 -> 0*4+0=0
    # c01: k=1,i=0 -> 1*4+0=4
    # c10: k=0,i=1 -> 0*4+1=1
    # c11: k=1,i=1 -> 1*4+1=5
    c_idx = (0, 4, 1, 5)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    # Process (A12, B2): C top-right 2x2 contribution to cols 0,1
    a_idx = blocks_A['A12']
    b_idx = blocks_B['B2']
    c_idx = (0, 4, 1, 5)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    # Process (A21, B1): C bottom-left 2x2 contribution to cols 0,1
    a_idx = blocks_A['A21']
    b_idx = blocks_B['B1']
    # C rows 2,3; cols 0,1 -> W indices:
    # c00: k=0,i=2 -> 0*4+2=2
    # c01: k=1,i=2 -> 1*4+2=6
    # c10: k=0,i=3 -> 0*4+3=3
    # c11: k=1,i=3 -> 1*4+3=7
    c_idx = (2, 6, 3, 7)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    # Process (A22, B2): C bottom-right 2x2 contribution to cols 0,1
    a_idx = blocks_A['A22']
    b_idx = blocks_B['B2']
    c_idx = (2, 6, 3, 7)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    # Now cols 2,3
    blocks_B_23 = {
        'B1': (2, 3, 7, 8),   # B[0,2], B[0,3], B[1,2], B[1,3]
        'B2': (12, 13, 17, 18) # B[2,2], B[2,3], B[3,2], B[3,3]
    }
    
    a_idx = blocks_A['A11']
    b_idx = blocks_B_23['B1']
    c_idx = (0, 4, 1, 5)  # same C positions, but these contribute to cols 2,3
    # Wait, I need to be careful. The C output for cols 2,3 has different W indices.
    # c00 for cols 2,3: k=2,i=0 -> 2*4+0=8
    # c01 for cols 2,3: k=3,i=0 -> 3*4+0=12
    # c10 for cols 2,3: k=2,i=1 -> 2*4+1=9
    # c11 for cols 2,3: k=3,i=1 -> 3*4+1=13
    c_idx = (8, 12, 9, 13)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    a_idx = blocks_A['A12']
    b_idx = blocks_B_23['B2']
    c_idx = (8, 12, 9, 13)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    a_idx = blocks_A['A21']
    b_idx = blocks_B_23['B1']
    # C rows 2,3; cols 2,3:
    # c00: k=2,i=2 -> 2*4+2=10
    # c01: k=3,i=2 -> 3*4+2=14
    # c10: k=2,i=3 -> 2*4+3=11
    # c11: k=3,i=3 -> 3*4+3=15
    c_idx = (10, 14, 11, 15)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    a_idx = blocks_A['A22']
    b_idx = blocks_B_23['B2']
    c_idx = (10, 14, 11, 15)
    col = add_strassen_block(a_idx, b_idx, c_idx, col)
    
    # Now col 4 (2x2 @ 2x1 for each of the 4 pairs)
    blocks_B_4 = {
        'B1': (4, 9),    # B[0,4], B[1,4]
        'B2': (14, 19)   # B[2,4], B[3,4]
    }
    
    a_idx = blocks_A['A11']
    b_idx = blocks_B_4['B1']
    # C col 4, rows 0,1:
    # c00: k=4,i=0 -> 4*4+0=16
    # c10: k=4,i=1 -> 4*4+1=17
    c_idx = (16, 17)
    col = add_schoolbook_2x2_1(a_idx, b_idx, c_idx, col)
    
    a_idx = blocks_A['A12']
    b_idx = blocks_B_4['B2']
    c_idx = (16, 17)
    col = add_schoolbook_2x2_1(a_idx, b_idx, c_idx, col)
    
    a_idx = blocks_A['A21']
    b_idx = blocks_B_4['B1']
    # C col 4, rows 2,3:
    # c00: k=4,i=2 -> 4*4+2=18
    # c10: k=4,i=3 -> 4*4+3=19
    c_idx = (18, 19)
    col = add_schoolbook_2x2_1(a_idx, b_idx, c_idx, col)
    
    a_idx = blocks_A['A22']
    b_idx = blocks_B_4['B2']
    c_idx = (18, 19)
    col = add_schoolbook_2x2_1(a_idx, b_idx, c_idx, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
