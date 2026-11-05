"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Returns factor matrices U, V, W with integer entries that exactly
    decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    
    We use a hierarchical block approach:
    - Split the 4x4 matrix A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    - Split the 4x7 matrix B into 2x2 blocks (with 7 columns split as 4+3):
      B = [[B11(2x4), B12(2x3)], [B21(2x4), B22(2x3)]]
    
    The product C = A*B has dimensions 4x7.
    C = [[C11(2x4), C12(2x3)], [C21(2x4), C22(2x3)]]
    
    Each C block is computed as sum of products of A blocks and B blocks:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    We use Strassen's algorithm for the 2x2 block multiplications.
    Strassen's algorithm uses 7 multiplications of 2x2 matrices instead of 4.
    
    For each pair of 2x2 blocks (A_block, B_block), we compute:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    However, for our case, the B blocks have different sizes (2x4 and 2x3).
    We need to handle this carefully.
    
    Let's use a simpler approach: schoolbook for the 2x2 blocks but with
    sharing of terms where possible.
    
    Actually, let's just implement the schoolbook decomposition correctly
    and see if we can find a better approach.
    
    For now, let's implement the schoolbook decomposition with rank 112.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    rank = n * m * p  # 112
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 28 x 112
    W = [[0] * rank for _ in range(p * n)]  # 28 x 112
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
