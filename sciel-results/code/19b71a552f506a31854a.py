"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2_decomposition():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U, V, W matrices for rank-7 decomposition.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    
    Strassen's 7 products for (A*B) where A,B are 2x2:
    Let A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    
    M1 = (a11+a22)*(b11+b22)
    M2 = (a21+a22)*b11
    M3 = a11*(b12-b22)
    M4 = a22*(b21-b11)
    M5 = (a11+a12)*b22
    M6 = (a21-a11)*(b11+b12)
    M7 = (a12-a22)*(b21+b22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U maps input A indices (i,j) to Strassen terms
    # V maps input B indices (j,k) to Strassen terms  
    # W maps output C indices (i,k) to Strassen terms
    
    # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[i*2+k]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    W[0][0] = 1  # C11
    W[3][3] = 1  # C22
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    W[2][1] = 1  # C21
    W[3][3] = -1 # C22
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1 # b22
    W[0][1] = 1  # C12
    W[3][3] = 1  # C22
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1 # b11
    W[0][0] = 1  # C11
    W[2][1] = 1  # C21
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    W[0][0] = -1 # C11
    W[0][1] = 1  # C12
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1  # a21
    U[0][5] = -1 # a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    W[3][3] = 1  # C22
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1  # a12
    U[3][6] = -1 # a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    W[0][0] = 1  # C11
    
    return U, V, W

def _build_4x4_4x7_strassen():
    """
    Build rank-64 decomposition for 4x4 * 4x7 using Strassen.
    
    Partition A (4x4) into four 2x2 blocks:
    A = [[A11, A12],
         [A21, A22]]
    
    Partition B (4x7) into 2x2 and 2x1 blocks:
    B = [[B11, B12],
         [B21, B22]]
    where B11, B21 are 2x4 and B12, B22 are 2x3
    
    Actually, let's think more carefully. We want to use Strassen for 2x2*2x2.
    For 4x4 * 4x7, we can split into:
    - Left half of B: 4x4 (columns 0-3)
    - Right half of B: 4x3 (columns 4-6)
    
    For the 4x4 * 4x4 part, use Strassen recursively.
    For the 4x4 * 4x3 part, use schoolbook or a variant.
    
    Let me reconsider. The standard approach:
    Split A into 2x2 blocks of size 2x2.
    Split B into 2x2 blocks of size 2x4 and 2x3.
    
    Actually, let's use a cleaner approach:
    Use Strassen for the 2x2*2x2 subproblems.
    
    For 4x4 * 4x7:
    Write A as 2x2 blocks of 2x2 matrices.
    Write B as 2x2 blocks where left column is 2x4 and right is 2x3.
    
    C = A*B has 2x2 blocks of 2x4 and 2x3.
    
    Each block C_ij = sum_k A_ik * B_kj
    
    For the 2x4 parts (j=0):
    C_i0 = A_i0 * B_00 + A_i1 * B_10
    where A_i0, A_i1 are 2x2, B_00, B_10 are 2x4
    
    This doesn't directly fit Strassen which is for 2x2*2x2.
    
    Alternative: Use Strassen on 2x2*2x2, and handle the extra columns separately.
    
    Let me try a different decomposition. Split B into B_left (4x4) and B_right (4x3).
    C_left = A * B_left (4x4)
    C_right = A * B_right (4x3)
    
    For C_left, use recursive Strassen: rank = 7^2 = 49
    For C_right, use schoolbook: rank = 4*4*3 = 48
    Total: 97
    
    That's worse than 112? No, 97 < 112. But we want 64.
    
    Let me think about the 64-rank construction.
    
    Actually, for 4x4*4x7, one approach:
    Use Strassen for 2x2*2x2, applied to the 4x4*4x4 part (rank 49).
    For the 4x4*4x3 part, we can use a 7-rank scheme for 2x2*2x3?
    
    Hmm, this is getting complex. Let me try a simpler approach first:
    Just use the schoolbook and see if we can verify it works, then try to optimize.
    """
    n, m, p = 4, 4, 7
    
    # For now, use schoolbook as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
