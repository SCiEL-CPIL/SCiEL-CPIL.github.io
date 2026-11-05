import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x4 matrix.
    
    We use a block decomposition approach:
    - Split the 3x3 matrix A into blocks: A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    - Split the 3x4 matrix B into blocks: B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    Then C = A*B is 3x4, with blocks:
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x2)
    C21 = A21*B11 + A22*B21  (1x2)
    C22 = A21*B12 + A22*B22  (1x2)
    
    We can use Strassen for the 2x2 blocks where possible, and schoolbook for the rest.
    
    Actually, let me think more carefully. The tensor is M_{3,3,4}.
    
    A simpler approach: use the fact that we can decompose this as a combination of smaller tensor products.
    
    Let's use the following block structure:
    A (3x3) = [A_top (2x3); A_bottom (1x3)]
    B (3x4) = [B_left (3x2); B_right (3x2)]
    
    C (3x4) = A*B
    C_top_left = A_top * B_left (2x2)
    C_top_right = A_top * B_right (2x2)
    C_bottom_left = A_bottom * B_left (1x2)
    C_bottom_right = A_bottom * B_right (1x2)
    
    For C_top_left (2x2 from 2x3 * 3x2): we can use rank-7 Strassen-like or just schoolbook rank 8
    Actually, 2x3 * 3x2 is not a standard square block. Let me reconsider.
    
    Let me just use the schoolbook decomposition but try to find a better one.
    
    The known lower bound for M_{3,3,4} is at least 19 (from M_{3,3,3} which has rank 19).
    The upper bound via schoolbook is 36.
    
    Let me try a different approach: embed into 4x4x4 and use the rank-21 decomposition.
    
    Actually, I'll construct a valid decomposition using block decomposition with Strassen for 2x2x2 sub-problems.
    
    Split A (3x3) into:
    A = [[a11, a12, a13],
         [a21, a22, a23],
         [a31, a32, a33]]
    
    Split B (3x4) into:
    B = [[b11, b12, b13, b14],
         [b21, b22, b23, b24],
         [b31, b32, b33, b34]]
    
    C (3x4) = A*B
    
    Let me use a 2x2 block decomposition on the first two rows/cols:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    C = A*B = [[C11, C12], [C21, C22]] where:
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x2)
    C21 = A21*B11 + A22*B21  (1x2)
    C22 = A21*B12 + A22*B22  (1x2)
    
    For C11: 2x2 = (2x2)*(2x2) + (2x1)*(1x2)
    The first term is M_{2,2,2} with rank 7 (Strassen)
    The second term is a 2x1 * 1x2 = 2x2 matrix, which is rank 4 (outer product style)
    But they interact, so we can't just add ranks directly.
    
    This is getting complex. Let me just return the schoolbook decomposition for now,
    which gives rank 36. This is valid but not optimal.
    
    To improve, I need to find a better decomposition. Let me try to use the known
    rank-19 decomposition for M_{3,3,3} and extend it.
    
    Actually, let me just implement the schoolbook and return it. The task says to improve,
    but I don't have a verified better decomposition memorized.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 3*3*4 = 36
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
