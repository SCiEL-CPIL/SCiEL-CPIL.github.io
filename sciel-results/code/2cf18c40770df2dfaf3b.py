"""Strassen-based rank-98 decomposition for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    
    Strategy: The 4x4x8 multiplication splits into two independent 4x4x4 blocks
    (output columns 0-3 and 4-7). Each 4x4x4 block can be done in rank 49 using
    a known optimal decomposition. Total rank = 2 * 49 = 98.
    
    We use the explicit rank-49 scheme for 4x4x4 based on Strassen's algorithm
    applied to 2x2 blocks, where each 2x2x2 block uses rank 7 (Strassen).
    For 4x4x4: split into 2x2 blocks -> 4 sub-multiplications of 2x2x2,
    but Strassen reduces the 2x2x2 from rank 8 to rank 7.
    Actually, the standard Strassen for 4x4x4 gives rank 2^3 * 7 = 56, 
    but the known optimal is 49. Let me use a known rank-49 scheme.
    
    Actually, let me use the simpler approach: Strassen recursion on 4x4x4.
    4x4x4 -> split into 2x2 blocks. The tensor for 4x4x4 can be decomposed
    using the Kronecker product of the 2x2x2 Strassen tensor.
    Rank of 2x2x2 Strassen = 7.
    Rank of 4x4x4 via Kronecker = 7 * 7 = 49.
    
    This is the key insight: the 4x4x4 multiplication tensor is isomorphic to
    the Kronecker square of the 2x2x2 multiplication tensor (under appropriate
    basis changes), and Strassen's rank-7 decomposition of 2x2x2 gives a
    rank-49 decomposition of 4x4x4.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Build the rank-49 decomposition for 4x4x4 using Kronecker of Strassen 2x2x2
    # Strassen 2x2x2: U2 (4x7), V2 (4x7), W2 (4x7)
    # Index conventions for 2x2x2:
    #   A is 2x2, indexed by (i,j) with i in {0,1}, j in {0,1}
    #   B is 2x2, indexed by (j,k) with j in {0,1}, k in {0,1}
    #   C is 2x2, indexed by (k,i) with i in {0,1}, k in {0,1}
    #   U[i*2+j, t], V[j*2+k, t], W[k*2+i, t] for t in range(7)
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U2: rows indexed by (i,j) -> i*2+j, cols by Strassen term t
    # For term t, U2[i*2+j, t] = coefficient of A[i,j] in the t-th product
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)*(B00+B11)
    U2[0][0] = 1  # A00
    U2[3][0] = 1  # A11
    V2[0][0] = 1  # B00
    V2[3][0] = 1  # B11
    W2[0][0] = 1  # C00 (k=0,i=0)
    W2[3][0] = 1  # C11 (k=1,i=1)
    
    # Term 1: (A10+A11)*B00
    U2[2][1] = 1  # A10
    U2[3][1] = 1  # A11
    V2[0][1] = 1  # B00
    W2[2][1] = 1  # C10 (k=1,i=0)
    W2[3][1] = -1 # C11 (k=1,i=1)
    
    # Term 2: A00*(B01-B11)
    U2[0][2] = 1  # A00
    V2[1][2] = 1  # B01
    V2[3][2] = -1 # B11
    W2[1][2] = 1  # C01 (k=0,i=1)
    W2[3][2] = 1  # C11 (k=1,i=1)
    
    # Term 3: A11*(B10-B00)
    U2[3][3] = 1  # A11
    V2[2][3] = 1  # B10
    V2[0][3] = -1 # B00
    W2[0][3] = 1  # C00 (k=0,i=0)
    W2[2][3] = 1  # C10 (k=1,i=0)
    
    # Term 4: (A00+A01)*B11
    U2[0][4] = 1  # A00
    U2[1][4] = 1  # A01
    V2[3][4] = 1  # B11
    W2[0][4] = -1 # C00 (k=0,i=0)
    W2[1][4] = 1  # C01 (k=0,i=1)
    
    # Term 5: (A10-A00)*(B00+B01)
    U2[2][5] = 1  # A10
    U2[0][5] = -1 # A00
    V2[0][5] = 1  # B00
    V2[1][5] = 1  # B01
    W2[3][5] = 1  # C11 (k=1,i=1)
    
    # Term 6: (A01-A11)*(B10+B11)
    U2[1][6] = 1  # A01
    U2[3][6] = -1 # A11
    V2[2][6] = 1  # B10
    V2[3][6] = 1  # B11
    W2[0][6] = 1  # C00 (k=0,i=0)
    W2[1][6] = 1  # C01 (k=0,i=1)
    
    # Now build the 4x4x4 decomposition via Kronecker product
    # The 4x4 matrices are viewed as 2x2 blocks of 2x2 submatrices.
    # A (4x4) = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B (4x4) = [[B00, B01], [B10, B11]] where each Bjk is 2x2
    # C (4x4) = [[C00, C01], [C10, C11]] where each Cki is 2x2
    #
    # C_ki = sum_{j} A_ij * B_jk (block matrix multiplication)
    # Each block product A_ij * B_jk is a 2x2 * 2x2 multiplication.
    #
    # The Kronecker structure: 
    #   U4 has rows indexed by (I, i, J, j) where (I,J) are block indices and (i,j) are within-block
    #   Actually, let's use the standard Kronecker product approach.
    #
    # For 4x4x4, we can think of it as:
    #   A is 4x4 = (2x2) x (2x2) in Kronecker sense
    #   The tensor T_4 = T_2 x T_2 (Kronecker product of tensors)
    #   So rank(T_4) <= rank(T_2)^2 = 7^2 = 49
    #
    # U4[a, t1*7+t2] = U2[a1, t1] * U2[a2, t2]
    # where a = a1*2 + a2, a1 = a // 2, a2 = a % 2
    # Wait, need to be careful with index mapping.
    #
    # For 2x2x2: U2 is 4x7, indexed by (i,j) -> row i*2+j
    # For 4x4x4: U4 is 16x49, indexed by (i,j) -> row i*4+j where i,j in {0,1,2,3}
    #
    # The Kronecker product mapping:
    #   i = i1*2 + i2, j = j1*2 + j2
    #   row4 = i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    #   col4 = t1*7 + t2
    #   U4[row4, col4] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col4 = t1 * 7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            row4 = (i1*2 + i2)*4 + (j1*2 + j2)
                            u_val = U2[i1*2 + j1][t1] * U2[i2*2 + j2][t2]
                            U4[row4][col4] += u_val
    
    for t1 in range(7):
        for t2 in range(7):
            col4 = t1 * 7 + t2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            row4 = (j1*2 + j2)*4 + (k1*2 + k2)
                            v_val = V2[j1*2 + k1][t1] * V2[j2*2 + k2][t2]
                            V4[row4][col4] += v_val
    
    for t1 in range(7):
        for t2 in range(7):
            col4 = t1 * 7 + t2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            row4 = (k1*2 + k2)*4 + (i1*2 + i2)
                            w_val = W2[k1*2 + i1][t1] * W2[k2*2 + i2][t2]
                            W4[row4][col4] += w_val
    
    # Now extend to 4x4x8: the output has 8 columns.
    # C is 4x8. We split into two 4x4 blocks: C[:, 0:4] and C[:, 4:8].
    # C[:, 0:4] = A * B[:, 0:4]
    # C[:, 4:8] = A * B[:, 4:8]
    #
    # For the first block (columns 0-3): use U4, V4, W4 directly.
    #   U has 16 rows (n*m = 4*4), V has 16 rows (m*p_block = 4*4), W has 16 rows (p_block*n = 4*4)
    #   But we need V to have m*p = 4*8 = 32 rows.
    #
    # For the second block (columns 4-7): same U, but V rows are shifted, and W rows are shifted.
    #   V_second[j*8 + k, col] = V4[j*4 + (k-4), col] for k in 4..7
    #   W_second[(k-4)*4 + i, col] = W4[(k-4)*4 + i, col] for k in 4..7
    #
    # Wait, the trace convention is W[k*n + i, r] where k is the column index of C (0..7) and i is the row index.
    # So for block 0 (k=0..3): W rows 0..15 correspond to W4
    # For block 1 (k=4..7): W rows 16..31 correspond to W4 shifted
    
    rank = 98
    U = [[0]*rank for _ in range(n*m)]   # 16 x 98
    V = [[0]*rank for _ in range(m*p)]   # 32 x 98
    W = [[0]*rank for _ in range(p*n)]   # 32 x 98
    
    # Block 0: columns 0-3 of C
    for col in range(49):
        for row in range(16):
            U[row][col] = U4[row][col]
            V[row][col] = V4[row][col]
            W[row][col] = W4[row][col]
    
    # Block 1: columns 4-7 of C
    for col in range(49):
        for row in range(16):
            U[row][col + 49] = U4[row][col]
            # V: for the second block, B rows are j*8 + k where k=4..7
            # V4 row index is j*4 + k' where k'=0..3
            # So V row j*8 + (k'+4) = V4[j*4 + k']
            V[row + 16][col + 49] = V4[row][col]
            # W: for the second block, W rows are (k'+4)*4 + i = 16 + k'*4 + i
            # W4 row index is k'*4 + i
            W[row + 16][col + 49] = W4[row][col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
