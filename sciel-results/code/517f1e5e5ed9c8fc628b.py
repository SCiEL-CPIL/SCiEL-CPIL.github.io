"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Returns factor matrices U, V, W with integer entries that exactly
    decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    
    We use a block-structured approach:
    - Split the 4x4 matrix A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    - Split the 4x7 matrix B into 2x2 blocks (with 7 columns split as 4+3 or 2+2+3):
      Let's split B into columns: first 4 columns and last 3 columns.
      Actually, let's split rows of B (which are 4 rows) into 2+2, and columns (7) into 4+3.
      B = [[B11(2x4), B12(2x3)], [B21(2x4), B22(2x3)]]
    
    The product C = A*B has dimensions 4x7.
    C = [[C11(2x4), C12(2x3)], [C21(2x4), C22(2x3)]]
    
    Each C block is computed as sum of products of A blocks and B blocks:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    This gives 8 terms, each being a product of a 2x2 block from A and a 2x{4 or 3} block from B.
    Each such product is a 2x{4 or 3} matrix.
    
    However, we need to express this as a rank decomposition. The standard schoolbook for 2x2 * 2x4 has rank 16, and 2x2 * 2x3 has rank 12. Total would be 4*16 + 4*12 = 64+48=112? That doesn't seem right.
    
    Let me reconsider. The total schoolbook rank is 4*4*7 = 112. So we need to find something better than 112.
    
    Actually, for matrix multiplication n x m times m x p, the minimum known rank for 4x4 * 4x7... Let me think about this differently.
    
    The tensor is (4*4)*(4*7)*(7*4) = 16*28*28. Schoolbook rank is 4*4*7 = 112.
    
    Let me try a different approach. I'll construct the decomposition explicitly using the fact that we can sometimes reduce rank by sharing terms.
    
    For now, let me just implement the schoolbook decomposition correctly and see if I can improve it.
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
